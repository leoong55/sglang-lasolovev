# H200: независимый эксперимент TP против TP+PP

Цель — измерить decode при большом числе одновременно активных пользователей на восьми H200. Предыдущая конфигурация исключена из эксперимента: CP/DCP не включаются ни в одном плече. Код образа остаётся форком нашей ветки, как было запрошено; её профиль запуска не используется.

Оба плеча используют один image digest, одинаковые веса, TileLang для prefill и decode, raw FP8 KV и одинаковые scheduler-патчи. Вывод о преимуществе или недостатке PP делается после измерения. Нельзя менять backend или формат весов вместе с топологией и приписывать всю разницу PP.

## Профили

| Профиль | TP × PP | EP на стадии | Назначение |
|---|---|---|---|
| `tp8-decode` | 8 × 1 | 8 | Контроль без PP |
| `pp2-decode` | 4 × 2 | 4 | Промежуточная глубина PP; слои 39/39 |
| `pp4-decode` | 2 × 4 | 2 | Архивная раскладка слоёв 21/20/20/17 |
| `tp8-archive` | 8 × 1 | 8 | Контроль с mixed chunk и HiCache |
| `pp4-archive` | 2 × 4 | 2 | Рецепт архива с mixed chunk и HiCache |

Общие начальные значения: context500000, mem-fraction0.90, chunk4096, FP8 KV, TileLang, C80. Чистые decode-профили отключают HiCache, mixed chunk и speculation. Prefill graphs отключены, decode graphs включены. Локальный graph bucket покрывает microbatch: при C80 это 80/40/20 для PP1/2/4. PP штатно меняет работу CPU overlap scheduler; это часть реализации PP. Предложенные разбиения слоёв ещё не доказаны оптимальными.

Архивные профили добавляют `hicache-size187`, `write_back`, `direct`, `layer_first`, mixed chunk. 187 GiB на ранг — почти 1.5 TiB на восемь рангов только для HiCache: при недостатке RAM явно уменьшить `--hicache-size` одинаково в обоих плечах. L3 не включён: архив требует реальное локальное хранилище. Максимальный контекст и доля памяти — настройки для проверки, не обещание вместимости. Допускается одинаково уменьшить `--context` / `--mem-fraction` после замера свободного пула.

W4AFP8/Humming поддерживается по умолчанию; для официальных FP8 весов указать `--weights fp8 --model-pvc <реальное имя PVC>`. Они будут смонтированы в `/mnt/model-pvc-fp8`, квантование и MoE backend выбираются штатно по checkpoint. Внутри одной пары формат весов одинаковый.

## Патчи

[Подробный аудит](AUDIT.md) объясняет каждую правку. В этой ревизии перенесены идеи патчей 3–7, включая исправления 4/6. Патчи 1/2 относятся к отсутствующим реализациям Flash/mHC и KPool; у полного GLM проверяется его собственный PP-путь, а native DSA indexer уже содержит chunking logits.

| Опция renderer | Переменная | Действие |
|---|---|---|
| `--admit` | `SGLANG_ENABLE_H200_ADMIT_FULL_NEED=1` | Полная потребность по всем PP-микробатчам с дедупликацией и учётом страниц |
| `--park` | `SGLANG_ENABLE_H200_PARK_CHUNKED_PREFILL=1` | Парковка незаконченного prefill при давлении на KV и наличии decode |
| `--short-bypass 512` | `SGLANG_H200_SHORT_BYPASS_TOKENS=512` | Общий дополнительный бюджет коротких prefill на батч |
| `--skip` | `SGLANG_ENABLE_H200_SKIP_NOT_FITTING=1` | Продолжение поиска после отказа конкретному запросу |
| `--patches` | Все четыре выше | Общий режим для сравнения топологий |

Без опций scheduler-патчи выключены. `run-ab.sh` и YAML из image artifact включают все четыре в обоих плечах. FP8-патч выбирается флагами `--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --kv-cache-dtype fp8_e4m3`. Он не зависит от scheduler-переменных. Переменные из оригинального архива не являются псевдонимами новых.

Эта адаптация scheduler предназначена для полного GLM с native MLA без speculation, disaggregation, SWA/Mamba, LoRA и priority preemption. Полное резервирование намеренно консервативно: ждёт заявленный max_new_tokens, учитывает незавершённый prefill, страницы и уже выделенный KV. Оно может уменьшать admission. Short-bypass и skip могут увеличивать ожидание крупных запросов; mixed-load замер должен учитывать оба класса.

## Образ и проверка на H200

Workflow `GLM53 H200 archive audit image` публикует отдельный immutable tag `ghcr.io/leoong55/sglang-lasolovev:glm53-h200-ab-v1-<commit>` и artifact с digest и YAML. Использовать новый digest из текущего workflow: предыдущий `d9e42fa3d4f0` не содержит этих дополнений.

Базовый CUDA13 image закреплён digest. Установщик проверяет весь кумулятивный overlay и cancellation-тесты. CPU CI проверяет scheduler, профили, контроль сравнения и выбор CUDA TileLang dispatch. CPU проверки и сборка образа **не подтверждают компиляцию или точность ядра на H200**. TileLang компилирует ядра при запуске на GPU.

Перед inference выполнить внутри нового образа на целевом GPU:

```bash
python3 /opt/glm53-h200-ab/h200/check_tilelang_gpu.py
```

Проверка выполняет настоящий raw KV writer, FP8 attention с разными числом голов/длиной запроса, tail64/0, маскированные indices и CUDA graph replay. Сравнение с FP32 reference использует те же квантованные входы: это тест ядра, не качества модели. Затем необходимы короткая генерация полного GLM, проверка качества относительно BF16/валидированного baseline и повторные complete/abort циклы с возвратом KV/слотов. Эти GPU проверки при подготовке не выполнялись.

Для Harbor распаковать source artifact и запустить `bash build.sh --push`. Веса в образ не включаются. Отдельный Deployment/Service называется `sglang-glm53-h200-ab`, namespace `inf-glm53`. Выделить ему восемь GPU; управляющий скрипт работает только с этим Deployment. Compile cache хранится в отдельном pod-local каталоге и не зависит от старого compile-cache PVC.

## Запуск чистого decode A/B

```bash
python3 render.py --image "$IMAGE" --profile tp8-decode --patches > A.yaml
python3 render.py --image "$IMAGE" --profile pp4-decode --patches > B.yaml
python3 render.py --image "$IMAGE" --profile pp4-archive --patches > archive.yaml

python3 bench_decode.py prepare \
  --tokenizer /mnt/model-pvc-w4fp8 \
  --concurrency 80 --input-tokens 8192 --output-tokens 2048 \
  --seed 12345 --output decode-c80-8k.json

IMAGE='ghcr.io/leoong55/sglang-lasolovev:glm53-h200-ab-v1-<commit>@sha256:<digest>' \
DATASET="$PWD/decode-c80-8k.json" \
RESULTS="$PWD/results-c80-8k" \
bash run-ab.sh
```

Для FP8 передать `WEIGHTS=fp8 MODEL_PVC=<имя>` и подготовить dataset его tokenizer. `PP_PROFILE=pp2-decode` выбирает промежуточный PP. `CONTEXT` и `MEM_FRACTION` одинаково меняют оба плеча. Адрес Service можно задать через `URL`.

Порядок A1→B1→B2→A2. Между плечами pod перезапускается, затем идёт прогрев той же волной и idle cache flush. Сохраняются dataset hash, manifests, image digest, фактические pod/server args, raw token counters, timings и логи. `compare.py` разрешает только изменения топологии и производных размеров microbatch/graphs. Отдельные режимы `admit`, `short`, `park`, `skip` предназначены для изменения одной scheduler-опции.

Главная метрика — output tok/s в общем интервале, где все C запросов уже декодируют; отдельно median ms/token, TTFT и хвосты клиентских stream gaps. Ошибки, retraction, restart или отсутствие достаточно длинного общего decode-окна делают точку невалидной для resident decode. Это нужно отдельно записать как ограничение вместимости/стабильности, а не скрыть из отчёта. C80 в очереди не равняется C80 на GPU.

Повторить C40/80/120/160, затем более длинные входы. В чистом decode scheduler-патчи могут почти не влиять: они обслуживают admission/prefill. Архивные mixed/cache профили замерять отдельно с длинным prefill и короткими arrivals; `bench_decode.py` их отклоняет. Для вывода «PP в целом хуже» одной раскладки и одного backend недостаточно; здесь проверяется конкретная реализация TP/PP на одинаковом TileLang-стеке.

## Исправление prefix-read и счётчик PP (2026-09-15)

В candidate `c3edb291c87f` raw FP8 allocator/writer и TileLang decode были согласованы,
но one-shot MHA при чтении сохранённого префикса вызывал scaled dequantizer.
Это приводило к `dim_quant: 576 != 656`. Префикс может появиться как из radix cache,
так и из предыдущего chunked prefill. Теперь reader выбирается по фактическим dtype
и ширине буфера: raw FP8 576 читается через pool в BF16, scaled 656 сохраняет
деквантование со scale-факторами. Исправление нужно и TP, и PP.

При PP2, `--max-running-requests 128`, `--pp-max-micro-batch-size 64` и штатном
`--pp-async-batch-depth 0` доступны два микробатча по 64. Существующий лог
`#running-req` показывает только текущий микробатч. PP0 и PP1 обрабатывают одни
и те же запросы на разных слоях, поэтому складывать их счётчики нельзя.
Новый `#pp-active-req` считает уникальные незавершённые запросы во всех слотах
локального PP-планировщика, включая prefill; очередь ожидания в него не входит.
Это снимок состояния на конкретной стадии, а не синхронизированная метрика всех GPU.
Prometheus `num_running_reqs` и `/v1/loads` продолжают показывать текущий батч.

Для проверки C128 генератор нагрузки тоже должен держать 128 одновременных запросов;
серверный лимит сам по себе нагрузку не создаёт. Decode graph buckets должны
покрывать microbatch 64; это отдельная настройка, не причина assertion выше.
После обновления повторить два последовательных запуска с одинаковыми префиксами
и длинный chunked prefill, затем C128. CPU регрессии проверяют выбор обоих layout,
ROCm, TBO, global PP layer id и дедупликацию 2×64. GPU smoke дополнен raw prefix
read с nonzero stage offset и повторяющимися KV indices; его нужно запустить на H200.
