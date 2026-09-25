# A1: проверка decode CUDA graphs для GLM-5.3 CP8/DCP4

Это **контрольная правка запуска**, а не доказанное исправление CP/DCP kernels.
Использует уже собранный образ v1. Пересборка и замена runtime overlay не нужны.
TP8/EP8/CP8/DCP4, FP8 KV, оба `flashmla_kv`, prefill eager, chunk 8192,
лимит запросов, память и фактический model PVC сохраняются.

В первом v1 оба вида CUDA graphs были выключены для проверки запуска. Такой
режим нельзя считать рабочим профилем производительности.

## Что показал decode trace

Прочитаны пять полных rank traces: TP 0, 1, 3, 4, 6. В них восемь DECODE
шагов с batch size 31, около 392–405 мс на шаг по CPU-аннотации. Prefill
в этом захвате отсутствует. Это измерения с включённым profiler, не независимый
throughput benchmark. На каждом доступном GPU — 42 352 kernel events, то есть
5 294 на шаг. CPU rank 0 зарегистрировал 225 203 torch operator events.

Суммарное время kernel events на каждом GPU за восемь шагов, мс:

| Rank | Custom all-reduce | NCCL all-gather + reduce-scatter |
|---|---:|---:|
| 0 | 2769.3 | 239.6 |
| 1 | 2622.4 | 33.7 |
| 3 | 2770.6 | 240.8 |
| 4 | 1343.6 | 1648.9 |
| 6 | 1365.9 | 1670.5 |

Эти строки нельзя складывать между GPU. Время collective kernel включает
ожидание остальных участников. Перемещение задержки между custom AR и NCCL
не позволяет объявить один из них неисправным. Сопоставленные custom AR на
доступных рангах завершаются почти одновременно при различном времени старта.

На rank 0 sparse FlashMLA decode kernel занимает 21.0 мс за восемь шагов
(дополнительно combine 5.6 мс); 75 MoE слоёв дают 1200 W4A8 GEMM kernel calls;
indexer paged MQA — 168 вызовов, то есть 21 на шаг, а attention — 78 на шаг.
Это согласуется с повторным использованием top-k в полной GLM-5.3. Нет
признака, что наш backport заставил все 78 слоёв повторять indexer.

Гипотеза A1: eager host dispatch и рассинхронизация рангов удерживают GPU
в ожидании. Full decode graphs обходят большую часть повторного Python/torch
dispatch. Graph mode также меняет выбор и способ запуска коммуникаций;
успех A1 ещё не отделяет эти эффекты друг от друга.
Full graph runner, CP decode attention TP и DSA decode metadata присутствуют
в закреплённом исходнике; GPU capture/replay этой комбинации здесь не проверен.

## Применить A1

Остановить генератор нагрузки и дождаться завершения текущих запросов. Этот
Deployment использует Recreate: смена аргументов перезапустит модель.

В каталоге этого комплекта:

```bash
kubectl -n inf-glm53 get deployment sglang-glm53-dcp4 -o json > deployment-before-a1.json
python3 make_decode_graph_patch.py deployment-before-a1.json --output-dir graph-a1
kubectl -n inf-glm53 patch deployment sglang-glm53-dcp4 --type=json --patch-file graph-a1/enable.json
kubectl -n inf-glm53 rollout status deployment/sglang-glm53-dcp4 --timeout=120m
```

Наблюдать старт в другом терминале:

```bash
kubectl -n inf-glm53 logs -f deployment/sglang-glm53-dcp4 -c sglang
```

Генератор меняет только decode graph flags в **фактическом** Deployment;
не возвращает старое имя model PVC из шаблона v1. JSON Patch проверяет UID,
контейнер, образ и прежние args перед записью. Файлы snapshot/patch могут
содержать локальные параметры запуска — хранить у себя.

Ручной эквивалент изменения аргументов:

```yaml
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

Удалить прежнюю пару `--cuda-graph-backend-decode disabled`, а также прежние
decode graph size flags, если они были. Prefill backend остаётся disabled.
Применённые серверные `--cuda-graph-config` и legacy disable flags могут
перекрывать этот выбор; генератор явно откажется при таком конфликте.

После запуска в **Decode batch** должна появиться строка `cuda graph: True`.
Readiness сама по себе graph replay не доказывает. При ошибке capture сохранить
traceback со всех рангов и выполнить откат; при тихом eager fallback сохранить
startup args и строки decode. Нужен также короткий осмысленный ответ модели,
поскольку увеличение throughput не проверяет численную корректность.

```bash
kubectl -n inf-glm53 patch deployment sglang-glm53-dcp4 --type=json --patch-file graph-a1/rollback.json
```

Откат также проверяет ожидаемые args и не затрёт более позднюю ручную правку.

## Короткое сравнение до/после

Запускать из benchmark pod с установленным vLLM 0.26.0. Сначала тот же тест
на v1, затем на A1. Во втором запуске заменить только result-filename.
Измерять без включённого profiler, после JIT-прогрева; cold TTFT оценивать отдельно.

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://sglang-glm53-dcp4.inf-glm53.svc:8080 \
  --endpoint /v1/chat/completions \
  --model GLM-5.3 \
  --tokenizer PhalaCloud/GLM-5.3-W4AFP8 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 60000 \
  --prefix-repetition-suffix-len 15000 \
  --prefix-repetition-output-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --request-rate inf --max-concurrency 1 --num-prompts 4 --num-warmups 1 \
  --ignore-eos --temperature 0.3 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":true}}' \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --save-result --save-detailed --result-dir /results \
  --result-filename glm53-v1-c1-out128.json
```

После успешного c1 повторить с concurrency 32, num-prompts 32 и другим именем
результата. Параллельно сохранить Decode batch строки. Для настоящего c40 потом
отдельно поднять max-running-requests и graph max_bs/последний bucket до 40.
Серверный лимит 32 при клиентских 40 сам создаёт очередь; менять его внутри A1
означало бы сравнивать разные нагрузки.

## Prefill: что проверить следующим захватом

Новые логи содержат CP ранги 0–7, chunk 8192, обычные значения 10–11.5k токен/с
и cache hits 59904. Это не доказательство эффективности CP8, но и не выключенный
CP по одному графику. Метрика `input throughput` в metrics_reporter делит число
новых токенов на время с прошлого prefill report; между ними может выполняться
decode. Такие строки нельзя использовать как время одного attention kernel.
Одинаковые TTFT/TPOT на панели требуют проверки PromQL: ITL на другом графике
измеряется сотнями миллисекунд, не минутами.

Проверено по коду: CP interleave делит новые query tokens, CP собирает текущие K,
а DCP дополнительно собирает сохранённый prefix KV на каждом attention слое.
В пути `flashmla_kv` Q имеет форму `[local_query_tokens, 1, heads, dim]` и
используется sparse decode kernel. Это совместимый путь PR #36990, но его
скорость prefill не равнозначна специализированному `flashmla_sparse_q8`.
Интеграция последнего должна учитывать gathered KV layout и logical top-k
mapping; простой переключатель backend в v1 запрещён не случайно.

Для отделения CP splitting, prefix gather, indexer и attention нужен stage
profile. Перед отдельным длинным запросом на свободном сервере, из benchmark pod:

```bash
curl -fsS -X POST http://sglang-glm53-dcp4.inf-glm53.svc:8080/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/mnt/cache/profiles/glm53-stages-a1","profile_id":"glm53-stages-a1","profile_by_stage":true,"num_steps":3,"activities":["CPU","GPU"],"with_stack":false,"record_shapes":true,"merge_profiles":false}'
```

Затем выполнить один длинный запрос (можно тот же bench с num-prompts 1).
В штатном profiler v1 `profile_by_stage` пишет prefill и decode раздельно,
по три шага; если prefill короче, завершает его при переходе в decode.
Нужно дождаться окончания записи обоих этапов на всех рангах, а не только
ответа /start_profile. Эти тайминги не смешивать с результатами bench без profiler.
Сохранить сопутствующие Prefill batch логи, особенно cached/new tokens.

Экспортировать после завершения записи, с control-plane узла:

```bash
kubectl -n inf-glm53 exec deployment/sglang-glm53-dcp4 -c sglang -- \
  sh -c 'for f in /mnt/cache/profiles/glm53-stages-a1/*.json.gz; do gzip -t "$f" || exit 1; done'
kubectl -n inf-glm53 exec deployment/sglang-glm53-dcp4 -c sglang -- \
  tar -C /mnt/cache/profiles -cf - glm53-stages-a1 > glm53-stages-a1.tar
tar -tf glm53-stages-a1.tar
sha256sum glm53-stages-a1.tar
```

Трейсы уже сжаты по отдельности. Следить за ненулевым exit code экспортирующей
команды: незавершённый tar может содержать несколько читаемых рангов и оборваться
позже. Нужны все восемь рангов: текущие неполные данные не позволяют назвать
последнего участника каждого collective.

## Локальная проверка

Генерация и применение JSON Patch к копии Deployment проверены с исходным v1
манифестом и другим model PVC, включая откат и отказ при изменившихся args.
Python syntax проверен. Ни kubectl против кластера, ни CUDA graph capture/replay
на 8 H200 из этого окружения не выполнялись. Численная корректность DCP остаётся
отдельной обязательной проверкой перед рабочей эксплуатацией.
