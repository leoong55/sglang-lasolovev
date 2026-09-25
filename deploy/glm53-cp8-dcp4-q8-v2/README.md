# GLM-5.3 CP8/DCP4: Q8 sparse prefill v2

Цель — проверить специализированный Hopper FP8 sparse prefill на нагрузке
60k prefix + 15k suffix -> 1k output, 20 префиксов, concurrency 32.
Это кандидат для проверки на H200: работоспособность и ускорение на GPU ещё
не подтверждены. В этом окружении нет Docker/BuildKit и CUDA GPU.

## Собрать и заменить образ

Готовый архив содержит Dockerfile, установщик и overlay поверх исходного
образа `v0.5.19-latest-0bcd822377da`. Библиотеки CUDA/FlashMLA не обновляются.
Исходник закреплён на `0bcd822377da7b5718e674eaf9c870d349424dd1`.

```bash
sha256sum -c SHA256SUMS
docker build -t i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-q8-v2-0bcd822377da .
docker push i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-q8-v2-0bcd822377da
```

В **текущем** YAML Deployment заменить образ контейнера `sglang`:

```yaml
image: i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-q8-v2-0bcd822377da
```

В существующем `args` заменить значение prefill backend:

```yaml
- --dsa-prefill-backend
- flashmla_sparse_q8
```

Команда запуска `/opt/glm53-cp8-dcp4-v1/launch.py` сохранена в новом образе для
совместимости с твоим Deployment. PVC, mountPath и command менять не требуется.
Новый launcher разрешает также `flashmla_kv` prefill для сравнения на том же образе.

Проверить в том же `args` следующие значения; существующие пары заменять,
не добавлять дубликаты:

```yaml
- --max-running-requests
- '32'
- --chunked-prefill-size
- '8192'
- --prefill-decode-interval
- '1'
- --cuda-graph-backend-prefill
- disabled
- --cuda-graph-backend-decode
- full
- --cuda-graph-max-bs-decode
- '32'
- --cuda-graph-bs-decode
- '1'
- '2'
- '4'
- '8'
- '16'
- '32'
```

TP8/EP8/DP1, CP interleave, DCP4/ag_rs, CP decode attention TP,
`--kv-cache-dtype fp8_e4m3`, page 64, mem fraction 0.80 и
`--dsa-decode-backend flashmla_kv` остаются как в v1. HiCache и speculative
decoding не включать. Prefill graphs для CP+DCP требуют отдельной интеграции.
ENV `SGLANG_ENABLE_CP_V2=1` и `SGLANG_DSA_FUSE_TOPK=0` остаются как в v1.

Для изменения непосредственно в Kubernetes:

```bash
kubectl -n inf-glm53 edit deployment sglang-glm53-dcp4
kubectl -n inf-glm53 logs -f deployment/sglang-glm53-dcp4 -c sglang
```

Deployment использует Recreate; применять после завершения текущей нагрузки.
Проверить startup marker `glm53-cp8-dcp4-q8-v2`, выбранные backend'ы и
`cuda graph: True` в Decode batch. На первом prefill возможен JIT q8 kernel.
Сначала проверить короткий осмысленный ответ и одиночный длинный запрос,
затем выполнять c32. При traceback сохранить полный лог со всех рангов.

Откат: вернуть прежний image tag
`glm53-cp8-dcp4-v1-0bcd822377da` и prefill `flashmla_kv`.
Для сравнения q8 и kv достаточно менять только prefill backend на **образе v2**:
другие параметры, включая decode graphs, должны совпадать.

При необходимости экспорт уже собранного образа отдельно:

```bash
docker save i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-q8-v2-0bcd822377da | gzip -1 > glm53-q8-v2-image.tar.gz
sha256sum glm53-q8-v2-image.tar.gz
```

Этот большой экспорт не запускается автоматически после сборки.

## Что изменено в runtime

Поверх предыдущего backport изменён один файл `dsa_backend.py`:

1. Для prefill CP+DCP разрешён `flashmla_sparse_q8`; decode остаётся `flashmla_kv`.
2. Indexer возвращает логические top-k. Они отображаются через
   `dcp_page_table_1` в собранный prefix+current KV buffer. Обычные RAGGED offsets
   здесь неверны: planner хранит все префиксы, затем все новые токены, а не
   последовательности запросов целиком друг за другом.
3. Q8 использует этот собранный FP8 buffer с identity row mapping. Штатный
   fused gather/dequant/requant преобразует 656-байтовые packed rows в
   576-компонентные FP8 rows для native SM90 Q8KV8 kernel. В локальный
   persistent DCP pool по глобальным индексам не обращаемся.
4. Короткий остаток prefill менее CP-size, когда interleave CP не применяется,
   использует прежний `flashmla_kv` Q-gather/LSE путь. Выбор kernel и построение
   metadata используют одну функцию, поэтому fallback получает FlashMLA metadata.
5. Decode LSE, DCP ownership, KV allocator и gather самих префиксов не менялись.

Индексер и top-k reuse GLM-5.3 остаются прежними. Q8 не избавляет от передачи KV
префикса между DCP-рангами. На холодном CP-prefill v2 тоже читает packed KV из
общего буфера; численный результат не обязан совпадать бит-в-бит с исходным
non-DCP q8-путём, который преобразует текущие BF16 K напрямую.

## Что именно измеряет твой бенч

`bench-c32.sh` — исходный workload с изменённой на 32 конкуренцией:
300 запросов, 20 префиксов по 60000 токенов, suffix 15000, output 1000,
thinking включён, request-rate inf. Запускать из benchmark pod:

```bash
bash bench-c32.sh
```

Размеры не уменьшены для достижения красивого результата.
Смена backend не делает холодные 60000 токенов автоматически закэшированными.
Один warmup не гарантирует прогрев 20 разных префиксов. При полном reuse
20 префиксов это примерно 1.2 млн токенов начальной обработки плюс
4.5 млн suffix tokens на 300 запросов. Одновременные cache misses и eviction
могут увеличить реальную работу; chat template/токенизация также добавляют разницу.

Оценка резидентного KV для 32 запросов, до page rounding и служебного запаса:

| Условие | Оценка |
|---|---:|
| Все 20 префиксов общие и сохранены | 20*60000 + 32*(15000+1000) = 1 712 000 токенов |
| Полностью раздельные истории | 32*(60000+15000+1000) = 2 432 000 токенов |

Ранее в логе около 1.4 млн used соответствовали ~41% пула. Это указывает
примерно на 3.4 млн логических токенов, но точную capacity брать из startup log.
Также нужны временные буферы и CUDA graph memory; запас KV не гарантирует
отсутствие prefill OOM. Старые suffix cache entries могут занимать свободную
часть пула и вытесняться radix cache.

Клиентская concurrency 40 при серверном max-running-requests 32 даёт очередь
даже при свободной KV памяти. Для цели c32 выставляем 32 с обеих сторон.
31 запроса в Decode batch при одном проходящем prefill не означает потерю CP-ранга.

Оценивать: 300/300 успешных запросов, ошибки/OOM/retraction, throughput,
TTFT p50/p95/p99, ITL и cache hits. Cold TTFT и прогретый prefix-hit TTFT —
разные режимы. Нужный предел TTFT пока не задан; сам факт обслуживания c32
не означает приемлемую задержку. Сравнивать kv/q8 без включённого profiler,
после JIT-прогрева, при одинаковой нагрузке и сопоставимом состоянии cache.

## Проверки и воспроизводимость

Локально прошли 14 CPU host-тестов: предыдущие 9 проверок KV/LSE и 5 проверок
q8 routing, mixed prefix/current layout, CP request subsets, cold prefill,
sentinel indices, small-tail fallback и отказа при неправильном dtype.
Загружаются настоящие функции исходника через AST. CUDA slot translation,
FP8 repacking и attention kernel заменены CPU reference/mocks: это не GPU-тест.
В частности, точность FP8 GMMA на реальной GLM-5.3 и производительность c32
на H200 ещё не подтверждены.

Из архива (нужен PyTorch):

```bash
SGLANG_SOURCE_ROOT="$PWD/overlay" python3 -m unittest discover -s tests -p 'test_host*.py' -v
```

`runtime.patch` — cumulative diff от закреплённого upstream;
`v1-to-v2.patch` — только runtime delta от предыдущего коммита.
При сборке Docker их вручную применять не нужно: установщик проверяет SHA-256
всех базовых файлов и ставит overlay. Повторная проверка выполняется при старте.
`REVISION.json` фиксирует base, source commit/tree и границы проверки.

Из GitHub checkout создать тот же комплект вне репозитория:

```bash
python3 deploy/glm53-cp8-dcp4-q8-v2/make_bundle.py /tmp/glm53-cp8-dcp4-q8-v2
```
