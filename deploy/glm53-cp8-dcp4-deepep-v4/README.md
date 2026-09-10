# GLM-5.3 CP8/DCP4: DeepEP prefill v4

Кандидат на H200 поверх базы `0bcd822377da7b5718e674eaf9c870d349424dd1`.
Сохраняет cumulative исправления v1–v3. Новый путь предназначен для полной
GLM-5.3-W4AFP8 (78 слоёв), а не GLM-5.3-Flash.

**CUDA/H200 и Docker здесь недоступны.** Архив содержит исходники, cumulative
patch, Dockerfile и установщик. Бинарный image нужно собрать приведённой ниже
командой. GPU/NCCL/DeepEP-корректность и ускорение ещё не измерены.
Проверки на CPU подтверждают контракт маршрутизации и выбор ветки decode,
но не являются проверкой численной точности GPU-кернелов.

## Сборка и замена image

Из распакованной директории:

```bash
sha256sum -c SHA256SUMS
docker build -t i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-deepep-v4-0bcd822377da .
docker push i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-deepep-v4-0bcd822377da
```

В текущем YAML Deployment `sglang-glm53-dcp4` изменить только image:

```yaml
image: i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-deepep-v4-0bcd822377da
```

Прежний `command: python3 /opt/glm53-cp8-dcp4-v1/launch.py` поддерживается.
PVC, args и env берутся из вашего рабочего v3 YAML. Глобальный
`--moe-a2a-backend none` **оставить**: отдельный prefill-путь включён внутри
нового образа через `SGLANG_GLM53_DEEPEP_PREFILL=1`.

Также включён `deployment-image-patch.yaml` — strategic merge patch именно
для текущего Deployment. Альтернативно правке своего файла его можно применить:

```bash
kubectl -n inf-glm53 patch deployment sglang-glm53-dcp4 --type=strategic --patch-file deployment-image-patch.yaml
```

Это patch, не полный Deployment для `kubectl apply`.
Ни сборка, ни push, ни изменение кластера автоматически не выполняются.

База Dockerfile — исходный `v0.5.19-latest-0bcd822377da`, не образ v3.
Установщик проверяет исходные хеши, включая новые файлы. Сборка также проверяет
импорт legacy `deep_ep.Buffer` и CUTLASS W4AFP8 entrypoints; зависимости не
обновляются. Если исходному образу не хватает DeepEP, сборка завершится явной
ошибкой. Обновление DeepEP до произвольной latest-версии не предусмотрено.

## Что делает v4

На MoE-слоях и только при активном CP-prefill:

1. LayerNorm и router получают локальную часть чанка CP, без сборки hidden
   states всех восьми рангов.
2. DeepEP **normal, BF16 dispatch** доставляет токены владельцам выбранных
   экспертов. Используется существующий W4AFP8 CUTLASS normal adapter и те же
   локальные routed-веса; новый формат весов не вводится.
3. Combine возвращает результат в исходный порядок локальных CP-токенов.
   Routing weights применяются в adapter, routed scaling factor — один раз
   после combine. Shared expert добавляется после масштабирования routed-ветки.
4. Отдельный TP1 shared expert вычисляется локально. Его checkpoint-веса и
   scales загружаются штатными weight loaders в дополнительный модуль;
   оригинальный TP-sharded shared expert сохранён для decode.
5. Padding CP исключается из маршрутизации; выход снова дополняется нулями
   до прежней физической формы.

Убираются hidden-state all-gather/reduce-scatter вокруг 75 routed MoE-слоёв.
Вместо них выполняются dispatch/combine; коммуникации не исчезают. KV/indexer
CP/DCP gather и 3 dense-слоя используют прежний путь.

Decode, включая full CUDA Graph, проходит через прежние original experts,
shared weights, dispatcher и communicator. Глобальные MoE backend/mode не
переключаются между forward-вызовами. Небольшой non-CP extend также использует
старый путь. Все routed-веса общие для обоих вариантов, без их дублирования.

## Профиль и память

| Параметр | Значение |
|---|---|
| TP / EP / CP / DCP / DP / PP | 8 / 8 / 8 interleave / 4 ag_rs / 1 / 1 |
| Prefill / decode attention | flashmla_sparse_q8 / flashmla_kv |
| Chunked prefill | 8192 |
| KV / page size | fp8_e4m3 / 64 |
| Decode CUDA Graph | прежний full, capture sizes до 32 |
| Prefill CUDA Graph | disabled |
| Глобальный MoE A2A backend | none |
| Prefill routed MoE | DeepEP legacy normal, BF16 transport |
| HiCache / speculative / TBO / SBO / EPLB | выключены |

Сохранить `SGLANG_ENABLE_CP_V2=1`, `SGLANG_DSA_FUSE_TOPK=0`, прежние
mem-fraction-static и max-running-requests. Профиль рассчитан на одну ноду.

Дополнительная память: полная FP8-копия shared expert на каждом GPU плюс
DeepEP transport arena. Для 75 слоёв, hidden=6144 и intermediate=2048 это
примерно **2,64 GiB/GPU только новых shared weights**, плюс scales и arena.
Фактический объём weights/scales печатается при загрузке.
Буфер DeepEP создаётся при конструировании модели, **до профилирования KV**.
Поэтому KV-пул автоматически учитывает расход, но может стать меньше, чем в v3.
Процент ускорения или неизменная ёмкость KV не обещаются.

## Проверка запуска

```bash
kubectl -n inf-glm53 logs -f deployment/sglang-glm53-dcp4 -c sglang
```

В логе должны быть:

- `glm53-cp8-dcp4-deepep-v4: verified ... files`;
- `DeepEP prefill=1`;
- `DeepEP normal/BF16 prefill buffer initialized before KV sizing`;
- `additional prefill shared-expert weights/scales ... GiB per GPU`;
- при первом CP-prefill: `prefill DeepEP active on CP rank ...`.

Отсутствие последнего сообщения при коротком non-CP запросе нормально.
Ёмкость KV после старта сравнивать с v3; скрытого повышения mem-fraction-static
нет. Для оценки выигрыша использовать сохранённый v3 reference: 644,15 с,
465,73 output tok/s, mean TTFT 24,12 с, P95 TTFT 72,96 с. Повторять v3 не нужно.

## Откат

Вернуть только image:

```yaml
image: i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-q8-v3-0bcd822377da
```

В этом же v4 образе `SGLANG_GLM53_DEEPEP_PREFILL=0` до старта процесса отключает
новый путь, создание вспомогательного shared expert и DeepEP arena. Включение
флага в уже работающем процессе не поддерживается.

## Состав и проверки

- `runtime.patch`: cumulative diff от закреплённого upstream.
- `v3-to-v4.patch`: только новая функциональность.
- `overlay/`, `base-files.json`, `SHA256SUMS`: проверяемый установочный комплект.
- `REVISION.json`: точный commit/tree, база и статус аппаратной проверки.
- `tests/`: CPU checks CP/DCP/KV/LSE/Q8 и нового MoE-пути.

```bash
OMP_NUM_THREADS=1 SGLANG_SOURCE_ROOT="$PWD/overlay" python3 -m unittest discover -s tests -p 'test_host*.py' -v
```

Новые проверки исполняют реальные функции v4 с CPU-моками DeepEP/CUTLASS:
CP-порядок всех рангов, пустые ранги, poisoned padding, routing scale/shared
addition, загрузка всех shared weights/scales без alias, отказ при пропуске
веса, неизменный выбор decode и dense-веток. Установщик отдельно проверяется
на добавлении файлов, повторном запуске и откате после ошибки записи.

Для воспроизведения комплекта из GitHub checkout:

```bash
python3 deploy/glm53-cp8-dcp4-deepep-v4/make_bundle.py /tmp/glm53-cp8-dcp4-deepep-v4
```

При необходимости экспорт уже собранного image:

```bash
docker save i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-deepep-v4-0bcd822377da | gzip -1 > glm53-deepep-v4-image.tar.gz
```
