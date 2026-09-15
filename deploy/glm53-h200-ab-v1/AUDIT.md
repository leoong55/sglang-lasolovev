# H200.zip: аудит для полной GLM‑5.3 и нашего v9.14

Проверенная база: `219206f7db9f7ffb24dad2605edf72af62d4939e`, ветка `work/glm53-cancel-v9-14`, PR #18. Исходный upstream: `0bcd822377da7b5718e674eaf9c870d349424dd1`, v0.5.19. SHA256 входного H200.zip: `8539bd1c9988a72c8eaf1f596cb97863c8b8ca4052c10dbe321c900f47b9a6f2`.

Изучены все семь `patch_*.py`, обе правки `_not-applied`, их обоснования, таблица конфигураций и измерительные скрипты. Новый форк сохраняет всю базу v9.14. Из runtime меняются только `environ.py`, `schedule_policy.py`, `scheduler.py`; kernels, CP/DCP, Humming, DFlash, HiCache и отмена SSE наследуются из базы.

| Патч | Решение | Основание по коду |
|---|---|---|
| 1. pp-residual | Не переносить | Цель — `glm5_next.py`, mHC/Flash. Файла нет в нашем commit. Полная `GlmMoeDsaForCausalLM` наследует `DeepseekV2ForCausalLM` (`models/glm4_moe.py`), это другой контракт PP. Отсутствие residual в mHC-графе не доказывает дефект обычной модели. |
| 2. kpool-logits-chunk | Не переносить | `dsa_indexer_kpool.py` отсутствует. В используемом `dsa_indexer.py` `_should_chunk_mqa_logits` уже вызывается и есть цикл logits → topk по q-строкам. Это не абсолютная гарантия отсутствия OOM: целый K и одна строка logits по-прежнему требуют памяти. |
| 3. fp8-tilelang | Не переносить | Патч меняет gate, dispatch и ABI KV-пула, содержит SM120/188 SM и NoPE-специализацию Flash. На H200 используем `flashmla_sparse_q8` + `flashmla_kv` с FP8 KV. Смена сырой FP8-раскладки может сломать существующие kernels и DCP; одной правкой gate это не решается. |
| 4. admit-full-need | Отклонить эту реализацию | Правильная идея — учитывать все PP microbatches. Но `collect_inflight_reqs()` и `adder.can_run_list` могут пересекаться; сумма считает один запрос дважды. Для inflight учитывается только оставшийся output, а ещё не сделанный prefill — нет. Нет резерва page alignment; `_lp()` при любой ошибке возвращает 0. Проверка стоит до временной блокировки cached prefix, меняющей evictable budget. Это не точный allocator reservation и не доказанная защита от OOM. |
| 5. park-chunked-prefill | Перенесён как opt-in | При `rem_total_tokens <= 0` допускает существующую ветку парковки, только если есть runnable decode. Одиночный запрос и prefill-only batch сохраняют native escape. Значение chunk budget 0 само по себе не включает парковку. Не исправляет глобальный PP-резерв и не ускоряет decode kernel. |
| 6. short-bypass | Отклонить эту реализацию | Обещание суммарного bypass ≤ S неверно: при S=512, rem=-448 патч разрешает ещё input=512, получая rem=-960. `max(chunk_tokens_limit,S)` не учитывает уже потраченный bypass. Ветка `ignore_eos` при отключённом radix идёт отдельным путём. Длинная голова с OTHER всё ещё обрывает поиск короткого запроса. Для выпуска нужен отдельный ограниченный бюджет и проверка всех путей admission. |
| 7. skip-not-fitting | Перенесён с исправлением | Исходный `continue` оставляет `batch_is_full=True`, установленный выше; следующий проход снова останавливается. Исправленная версия сбрасывает этот флаг, но только если кандидат НЕ добавлен и `adder.budget_state()==CONTINUE`. Сохраняет native cleanup Mamba до продолжения. Не обходит исчерпание глобального KV/chunk/input budget или OTHER. |
| v1_tile / v1_headblock | Не переносить | Настройка sparse TileLang v1 под SM120 shared memory. Наш decode идёт через FlashMLA на SM90. Эти изменения не участвуют в данном пути. |

Новые переменные: `SGLANG_ENABLE_H200_PARK_CHUNKED_PREFILL=0/1` и `SGLANG_ENABLE_H200_SKIP_NOT_FITTING=0/1`. Обе по умолчанию 0, зарегистрированы в `Envs`. Имена из архива автоматически не включают новый код. Применять опции пока только в изолированных native профилях полной GLM без speculation, disaggregation, Mamba/SWA, LoRA и priority preemption. GPU-проверка их жизненного цикла ещё не выполнена.

Плата за skip: длинный запрос может ждать дольше, вплоть до starvation при непрерывном потоке мелких; просмотр очереди добавляет CPU/radix работу. Плата за park: KV остановленного prefill остаётся занят, а decode batch может быть небольшим. Ни один из них не является механизмом выгрузки активного decode KV в RAM.

## Что в расчётах и конфиге архива не переносится напрямую

- Полная GLM‑5.3 — MoE. Слово «плотная» в архиве можно понимать только как отсутствие гибридных linear-attention слоёв, но не как dense FFN. 78 — число слоёв, не активных миллиардов параметров.
- Измерения архива относятся к шести картам, TP2×PP3, другой модели/ветке. TP2×PP4 для восьми H200 — предлагаемый профиль, а не измеренный победитель.
- Таблица исходит из FP8 весов около 701.6 GiB, одинакового KV cost на 78 слоях и TP без DCP. Наш W4AFP8/Humming, DCP4 и producer-only index storage меняют исходные величины. Нельзя переносить «1.31 → 4.88 сессии» на наш пул.
- Раскладка `21,20,20,17` остаётся отдельным кандидатом из архива. Поправка автора на ~18.5 GiB MTP-головы не подтверждена для target-only W4 запуска без speculative worker. Баланс памяти и времени нужно снять на каждой PP-стадии; раскладка не объявляется оптимальной.
- PP4 требует CP off; upstream отключает overlap scheduling. DFlash с PP>1 запрещён в закреплённой базе. Эти ограничения сохранены.
- HiCache L2/L3 увеличивает повторное использование префиксов между ходами. Объём сохранённых сессий на CPU/диске не равен числу одновременно исполняемых decode-запросов на GPU.
- `hicache-size 187` на каждый из восьми rank занимает почти весь заявленный RAM-бюджет до служебных расходов. В адаптированном профиле — 96 GiB/rank; L3 не подключён без реального PVC, ключей и ёмкости. DCP+L3 в нашей базе отдельно запрещён.
- Архивные mem=0.90 и context=500000 не являются проверенными лимитами. Стартовый профиль использует mem=0.80 и context=131072. Context500000 доступен отдельной явной настройкой, после проверки model config и пула.
- Graph max16 не покрывает PP4 microbatch20 при C80. Renderer явно задаёт microbatch ceiling и его последний decode bucket. Число одновременно resident запросов нельзя выводить из client concurrency.

## Аудит инструмента измерения

`park-probe.py` записывает timestamp на каждую SSE data-строку, включая usage-only frame. Поэтому его разрывы нельзя без проверки трактовать как per-token ITL. Его профиль проверяет mixed workload, cache, очередь и admission; общий makespan смешивает prefill и decode. `llm-configs.sh` включает L3, зависит от локальных docker paths и стартовых defaults; в launch-команде нет явного `--kv-cache-dtype fp8_e4m3`, хотя расчёты опираются на FP8 KV. `run-ab.sh` передаёт это через окружение/таблицу и содержит удаление содержимого каталога L3; он не перенесён в K8s-форк.

Новый `bench_decode.py` использует фиксированные input token IDs, нулевую temperature, ignore_eos, точную длину output и одинаковый dataset hash. Поддерживает только native /generate без DFlash и без HiCache/mixed chunk. Общий decode-интервал начинается после 64 output tokens у самого позднего запроса и заканчивается перед последними 64 tokens самого раннего. Ошибки, неполные ответы, retraction, рестарт или отсутствие общего интервала ≥10 секунд делают результат недействительным. TTFT считается отдельно. SSE arrival gaps подписаны отдельно от ITL: coalesced events не превращаются в вымышленные отдельные token timestamps.

Это измерение реального serving decode с его scheduler/CPU/collectives, а не CUDA-only kernel benchmark. Одинаковый набор input IDs синтетический: он не проверяет качество, cache hit ratio или tool calling. Для итогового выбора нужен ещё отдельный повтор реальной агентной нагрузки и корректности длинного контекста.

## Проверки и границы

Локально исполняются реальные методы PrefillAdder/Scheduler, извлечённые AST, с fake allocator/worker boundaries. Проверяются default-off, runnable-decode guard, одиночный chunk, budget=0, cleanup и продолжение на следующем кандидате. Это CPU-контроль логики; он не моделирует все асинхронные PP/HiCache/KV события.

Workflow собирает cumulative overlay из точных commit, сверяет hashes, выполняет cancellation/ASGI тесты установленных файлов, парсит все новые native профили SGLang и повторяет scheduler tests внутри CUDA-образа. Публикуется отдельный candidate tag с SHA исходников и digest. H200 startup, PP-графы, numerical parity, KV lifecycle и производительность остаются неподтверждёнными до запуска на узле.
