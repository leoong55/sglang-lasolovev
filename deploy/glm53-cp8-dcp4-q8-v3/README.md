# GLM-5.3 CP8/DCP4 Q8 v3: меньше накладных расходов gather

Патч поверх q8-v2 оптимизирует подготовку CP/DCP KV gather на prefill.
База образа и исходников: `0bcd822377da7b5718e674eaf9c870d349424dd1`.
Предыдущая версия: `819468af5a3b2b490a0ab546f12f12069c6c108b`.
Основа DCP backport — PR #36990, head `c6aeb8b9d9128b816777e2b64cbf6603344e8fe1`.

Это кандидат для проверки на H200. Docker/BuildKit и CUDA GPU здесь недоступны:
готовый бинарный образ не собран, ускорение на GPU не измерено.
Архив содержит весь код, Dockerfile и установщик для сборки поверх исходного образа.

## Сборка

После распаковки перейти в каталог с Dockerfile:

```bash
sha256sum -c SHA256SUMS
docker build -t i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-q8-v3-0bcd822377da .
docker push i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-q8-v3-0bcd822377da
```

Сборка использует исходный образ `v0.5.19-latest-0bcd822377da`, устанавливает
весь cumulative overlay и проверяет SHA-256 каждого изменяемого файла.
Предварительно ставить v1/v2 или применять patch вручную не требуется.
Не подменять BASE_IMAGE на ранее исправленный образ: у него другие входные хеши.

## Изменение текущего YAML

В существующем Deployment `sglang-glm53-dcp4` изменить **только image** контейнера `sglang`:

```yaml
image: i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-q8-v3-0bcd822377da
```

Прежний command `python3 /opt/glm53-cp8-dcp4-v1/launch.py` поддерживается.
PVC, пути модели и кэша, ресурсы, args и env сохраняются из рабочего q8-v2 YAML.
Полный Deployment намеренно не включён: фактический model PVC должен оставаться
тем, с которого уже успешно работает модель.

Профиль этого сравнения:

| Параметр | Значение |
| --- | --- |
| Модель | Полная GLM-5.3-W4AFP8, 78 слоёв |
| TP / EP / CP / DCP / DP | 8 / 8 / 8 interleave / 4 ag_rs / 1 |
| Prefill / decode backend | flashmla_sparse_q8 / flashmla_kv |
| Chunked prefill | 8192 |
| KV / page | fp8_e4m3 / 64 |
| Decode graphs | full, прежние capture sizes до 32 |
| Prefill graphs | disabled |
| HiCache / speculative decoding | выключены |

Оставить `SGLANG_ENABLE_CP_V2=1` и `SGLANG_DSA_FUSE_TOPK=0` из текущего профиля.
Не менять mem-fraction-static, лимит одновременных запросов и параметры бенча
вместе с образом — иначе сравнение потеряет смысл.

После применения своего YAML:

```bash
kubectl -n inf-glm53 logs -f deployment/sglang-glm53-dcp4 -c sglang
```

Ожидаемый startup marker: `glm53-cp8-dcp4-q8-v3: verified 13 files`.
Launcher дополнительно печатает профиль CP8/DCP4 и выбранные backend'ы.
Откат — вернуть в том же YAML image
`i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-q8-v2-0bcd822377da`.
Args при откате не менять.

Для выгрузки уже собранного Docker image в отдельный большой архив:

```bash
docker save i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-q8-v3-0bcd822377da | gzip -1 > glm53-q8-v3-image.tar.gz
```

Этот экспорт не запускается автоматически.

## Изменения runtime

1. **DCP prefix layout готовится один раз на forward.** Planner преобразует
   CPU lengths в целые числа и проверяет выравнивание каждого префикса по DCP-size.
   Для выровненных packed FP8 префиксов не нужны padding, slice и cat по каждому
   запросу на каждом слое. Невыровненные префиксы и непакованные форматы сохраняют
   предыдущий generic путь; это не расширяет поддержку невыравненного allocator.
2. **Прямая запись собранного префикса.** Локальные строки выбираются в reusable
   buffer, выполняется прежний all-gather, затем одна rank/row перестановка пишет
   результат прямо в `dcp_kv_buffer`. Количество collective-вызовов и объём
   передаваемого KV не уменьшаются. Packed 656-байтовые строки передаются как uint8.
3. **CP interleave без тензора индексов.** Для DCP-профиля прежние
   arange/modulo/divide/index_select заменены перестановкой rank/row и обрезкой
   хвостового padding. Выход остаётся отдельным тензором; входной и транспортный
   буферы переиспользуются между слоями.
4. **Время жизни ограничено текущим batch и stream.** Ключи включают устройство,
   CUDA stream, группу и форму; для CP также dtype и logical/physical lengths.
   Работа на разных streams не использует один scratch buffer. KV values никогда
   не кэшируются между слоями. Новый слой каждый раз читает свой persistent pool.

Смысл stream-изоляции соответствует
[CUDA stream semantics в PyTorch](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-streams).
Существующие зависимости stream у потребителей KV сохраняются.

Патч не меняет ownership KV, page tables, Q8 repacking, численную логику attention,
LSE merge, scheduler, MoE/EP topology и decode. CP-путь без DCP остаётся прежним.
Повторный full-prefix all-gather на каждом слое остаётся; MoE collectives тоже
остаются. Поэтому это ограниченная оптимизация накладных расходов prefill,
а не обещание кратного ускорения или гарантированного TTFT при c32.

## Проверки

22 CPU host-теста проверяют реальные функции исходника через AST:
предыдущие 14 KV/LSE/Q8 проверок и 8 проверок prepared gather.
Среди новых проверок: DCP2/4/8 на всех рангах, пустые и разные по длине префиксы,
побайтовое совпадение с независимым эталоном и generic gather, смена слоя,
отсутствие alias у CP-результата, poisoned padding, noncontiguous input,
разные формы/streams/batches и отказ от fast path при невыравненном префиксе.
Collectives, CUDA kernels и stream IDs заменены CPU mocks: это не GPU/NCCL тесты.

Запуск из распакованного архива, нужен PyTorch с float8 dtype:

```bash
OMP_NUM_THREADS=1 SGLANG_SOURCE_ROOT="$PWD/overlay" python3 -m unittest discover -s tests -p 'test_host*.py' -v
```

Установщик отдельно проверен на исходных файлах pinned base, включая повторную
установку, verify-only, отказ при несовпадении исходника и rollback при ошибке
записи. Runtime diff проверен через git apply и сравнение SHA-256 overlay.
Полная точность модели, реальная асинхронная работа NCCL и производительность
v3 на H200 здесь не проверялись.

Текущий q8-v2 прогон повторять не требуется: он уже служит reference.
После замены образа сравнивать с ним тот же workload без включённого profiler.
Сохранить исходные c40, 300 запросов, 20 префиксов, 60000/15000/1000 и chunk8192.
Изменение очереди/TTFT оценивать вместе с cache hits и успешностью запросов.

## Состав и воспроизводимость

- `runtime.patch`: cumulative runtime diff от закреплённого upstream.
- `v2-to-v3.patch`: только пять runtime-файлов этой оптимизации.
- `overlay/`: 13 изменённых runtime-файлов с проверяемыми хешами.
- `REVISION.json`: base/source commit и tree, предыдущая версия, границы проверки.
- `base-files.json`, `SHA256SUMS`: контроль исходников и содержимого комплекта.
- `tests/`: CPU suite и исходный integration-тест backport.

Из GitHub checkout сформировать комплект вне репозитория:

```bash
python3 deploy/glm53-cp8-dcp4-q8-v3/make_bundle.py /tmp/glm53-cp8-dcp4-q8-v3
```
