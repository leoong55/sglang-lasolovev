# GLM53 H200 A/B — отдельный форк v9.14

Этот комплект сохраняет все изменения v9.14 и добавляет две независимые экспериментальные опции из H200.zip: парковку chunked prefill и исправленный пропуск невлезающей головы очереди. Обе выключены по умолчанию. Остальные архивные правки отклонены по результатам [аудита каждого патча](AUDIT.md).

Это candidate для восьми H200, а не подтверждённый ускоренный production image. GPU inference и измерения не выполнялись при подготовке. Ветка `work/glm53-h200-ab-v1` отделена от текущего выпуска.

## Образ и сборка

Workflow `GLM53 H200 archive audit image` выпускает:

- `ghcr.io/leoong55/sglang-lasolovev:glm53-h200-ab-v1-<12 символов commit>`;
- artifact `glm53-h200-ab-image-<commit>` с `image.json` (digest) и тремя YAML;
- checksummed source bundle `glm53-h200-ab-source-<commit>` для сборки в Harbor.

Image base закреплён на публичном SGLang v0.5.19 CUDA13 digest. Кумулятивный overlay восстанавливает всю нашу ветку, включая Humming и v9.14 cancellation. Сборка не скачивает веса и не пересобирает CUDA kernels. Docker RUN проверяет SHA исходников, Humming dependency, cancellation/ASGI, native parser профилей и scheduler logic. Совпадение кода не подменяет GPU-проверку.

Для Harbor распаковать source artifact и выполнить из каталога комплекта:

```bash
bash build.sh --push
```

Тег будет `i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-h200-ab-v1-<commit>`. Существующие теги не перезаписываются. `IMAGE` позволяет задать другой registry/tag. `GLM53_BASE_IMAGE` допускает совместимый закреплённый base; installer всё равно проверяет реальные файлы.

## Профили

| Профиль | Топология | Назначение |
|---|---|---|
| `tp8-dcp4-decode` | TP8/EP8, CP8 interleave, DCP4, PP1 | Контроль нашей топологии без DFlash, HiCache и prefill graphs |
| `pp4-decode` | TP2/EP2 × PP4, CP off, DCP1, partition21/20/20/17 | Второе плечо чистого serving decode |
| `pp4-archive` | Та же PP4; mixed chunk, HiCache96/rank, write_back | Адаптированный рецепт архива для отдельной mixed-load проверки |

Общие значения: W4AFP8/Humming, FP8 KV, FlashMLA, mem0.80, chunk4096, C80, context131072, native decode CUDA graphs. В PP4 max microbatch20 и graph bucket20; в TP8 graph bucket80. Разный размер локального batch — следствие топологии. PP требует отключения CPU overlap scheduler, upstream делает это сам. DFlash/CP ограничения для PP не снимаются.

Renderer создаёт отдельные Deployment и Service `sglang-glm53-h200-ab` в `inf-glm53`, на прежнем узле и существующих W4/compile-cache PVC. Плечи запускаются по очереди на одних восьми GPU. Нужно предварительно освободить этот узел от другого GPU-serving workload; скрипт управляет только lab Deployment. Сохранять исходный production YAML нужно в обычном процессе управления кластером.

```bash
python3 render.py --image "$IMAGE" --profile tp8-dcp4-decode > A.yaml
python3 render.py --image "$IMAGE" --profile pp4-decode > B.yaml
python3 render.py --image "$IMAGE" --profile pp4-archive --park --skip > archive.yaml
```

`IMAGE` должен содержать digest из artifact. Новые флаги `--park` и `--skip` меняют только соответствующие переменные; full-need и short-bypass не включены. Старый validated launcher доступен через `--glm53-profile`: для возврата к прежнему профилю передаются его прежние аргументы. Новые native профили не проходят через CP8-only wrapper.

## Чистое A/B decode

На машине с доступом к tokenizer PVC один раз создать токенизированную нагрузку. Требуются Python, aiohttp, PyYAML и transformers для `prepare`; для `run` tokenizer больше не нужен.

```bash
python3 bench_decode.py prepare \
  --tokenizer /mnt/model-pvc-w4fp8 \
  --concurrency 80 --input-tokens 8192 --output-tokens 2048 \
  --seed 12345 --output decode-c80-8k.json
```

Запускать следующий скрипт с хоста/клиентского pod, где доступны kubectl, эти Python-зависимости и адрес Service. Он выполняет A1→B1→B2→A2, рестарт между плечами, прогрев такой же волной, idle cache flush, измеренную волну и сохранение logs/pod JSON. Текущий рабочий Deployment скрипт не останавливает. Не использовать lab Service для другого трафика во время теста.

```bash
IMAGE='ghcr.io/leoong55/sglang-lasolovev:glm53-h200-ab-v1-<commit>@sha256:<digest>' \
DATASET="$PWD/decode-c80-8k.json" \
RESULTS="$PWD/results-c80-8k" \
bash run-ab.sh
```

При доступе через другой адрес задать `URL`; он должен вести на lab Service и переживать смену pod. Итоги сравниваются `compare.py`, сырые cumulative token counters и timestamps остаются в `requests.json`. Подготовить такие же отдельные серии C40, C80, C120, C160; для каждой renderer задаёт тот же server cap. Затем повторить с input32k и75k. Невозможность одновременного resident decode при длинном контексте — результат проверки ёмкости: нельзя выдавать queued C80 за 80 одновременно декодируемых запросов.

Главные показатели: output tok/s в общем decode-интервале, median per-request ms/token и хвосты stream gaps. TTFT печатается отдельно. Для каждого плеча сохранить pool sizes по rank, graph replay, GPU memory/utilization и значения retraction counters из логов. При errors/retraction/restart/коротком общем окне скрипт завершает тест с ошибкой. Интервал SSE не считается точным GPU ITL при coalescing.

## Проверка планировщика

В чистом resident decode патчи 5/7 обычно не имеют работы; ожидаемый результат — отсутствие ускорения. Их пользу измерять на отдельной mixed нагрузке:

1. Одна и та же PP4-конфигурация и image, сначала обе опции0.
2. Менять только `--park`, остальные параметры сохранять. Нужен наблюдаемый KV pressure и незавершённый chunk рядом с runnable decode.
3. Отдельная пара с изменением только `--skip`: большая не помещающаяся голова + короткие запросы. Измерять TTFT **обоих** классов, максимальное ожидание большого запроса, retractions и total makespan.
4. Только затем совместная проверка двух опций, включая cancellation, complete/abort и отсутствие утечки KV/слотов после опустошения очереди.

Полная пара на `pp4-archive` — это mixed/cache benchmark. `bench_decode.py` намеренно отклоняет такой профиль; его результаты нельзя включить в чистую таблицу decode. Архивный `park-probe.py` требует исправления SSE accounting перед использованием как ITL-измерителя.

## Отдельный FP8 вариант

Официальные FP8 веса по `/mnt/model-pvc-fp8` поддерживаются renderer, но имя реального PVC нужно указать явно:

```bash
python3 render.py --image "$IMAGE" --profile pp4-archive \
  --weights fp8 --model-pvc "$FP8_MODEL_PVC" --context 500000 > pp4-fp8.yaml
```

FP8 checkpoint выбирает свою quantization config и штатный backend; Humming W4 не навязывается. Context500000 требует соответствующего config модели и измеренного свободного пула. Не сравнивать этот вариант с W4 A и называть разницу эффектом PP или scheduler. Сначала провести TP8/DCP4↔PP4 на одинаковых FP8 весах отдельно. RAM/L3 session capacity не равна active decode concurrency.

## Проверки

```bash
python3 -m unittest discover -s deploy/glm53-h200-ab-v1/tests -v
```

CPU tests проверяют реальные scheduler methods через AST с fake внешними зависимостями, renderer и расчёт общей decode-фазы. Docker отдельно парсит CLI установленной версии. Numerical parity, PP/H200 graph replay, устойчивость mixed-load и выигрыш скорости подтверждаются только реальным GPU запуском.
