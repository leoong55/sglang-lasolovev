# GLM-5.3: prefill 16k, серия P0–P6

Исходная точка — `fcd40b5f1` (подготовленный 0.5.20 с накопленными патчами).
Серия реализована в исходниках и разбита на коммиты. На H200 проверены
CPU/CUDA gates и четыре нагрузочных профиля:2508 запросов без ошибок.
**Общее ускорение большинства экспериментальных путей не подтверждено;
Q-stream eager показал регрессию.** P6 и полная численная проверка cache-пути
остаются открытыми. Итогового подтверждённого GPU-профиля пока нет.
См. [промежуточные измерения](H200_INTERIM_20260926.md).
Новый образ не собирался; тесты выполнены source overlay поверх образа0.5.20.

| Патч | Реализация | Отключение / контроль |
|---|---|---|
| P0 | Причины admission, токены/контекст/доля продолжения/RID в JSON-журнале; CPU timings и неблокирующие CUDA events; диапазоны gather, KV convert, indexer/top-k, attention, MoE, draft append, HiCache | `SGLANG_GLM53_PREFILL_DIAGNOSTICS=0` |
| P1 | 16384, BCG 8192/16384, interval 1, slots 40, min-free-slots-delay 1; HTTP-прогрев до readiness; проверка реально созданного backend и buckets | `profile-control.json`; прогрев для отдельной диагностики выключается env `SGLANG_GLM53_PREFILL_WARMUP=0` |
| P2 | Один partial, полные соседние остатки; HRRN, максимум 128 admission-кандидатов; половина бюджета + заимствование + возврат долга; повторная проверка fullness | `--no-prefill-interleaving`; режим `upstream` вместо `adaptive` |
| P3 | Compact DeepGEMM logits, request-relative top-k, выровненные сбалансированные Q-порции; явный workspace MiB резервируется до KV и при post-capture resizing | `--glm53-dsa-indexer-mode legacy`; убрать `--glm53-dsa-logits-workspace-mib` для возврата 5% |
| P4 | Prefix KV остаётся локальным; одна конверсия на слой; 256 Q-строк/rank; top-k ownership через allocator read IDs; Triton compaction; FP32 LSE и token-axis reduce-scatter | `--glm53-dcp-prefill-mode kv-gather` |
| P5 | Один async MIN snapshot на отдельной Gloo-группе; применение в следующем проходе; immutable digest; контроль очередей; drain перед reset/idle/shutdown/другими drains | `--glm53-hicache-event-sync sync` |
| P6 | Buckets 4096/8192/12288/16384 с padding ≤25%; захват фиксированной Q-порции вместе с collectives, kernel и LSE combine; стабильные staging buffers | `--glm53-prefill-attention-graph off`; вернуть buckets 8192/16384 |

P2 опирается на механизмы [#40024](https://github.com/sgl-project/sglang/pull/40024),
[#39717](https://github.com/sgl-project/sglang/pull/39717) и regression scenario
[#35609](https://github.com/sgl-project/sglang/pull/35609). Это адаптация, не
дословный cherry-pick. Реальный PrefillAdder остаётся последней проверкой KV,
слотов и host loadback; после host miss второй partial запрещён повторно.
В режиме adaptive минимум — **логическая страница 256 токенов** для DCP4/page64.
Заимствование выдаёт долг; следующий конкурирующий проход сначала возвращает его.
Если допустимых соседей нет, продолжение получает полный доступный бюджет.
Незакэшированный остаток больше 16k не становится вторым partial.

P3 использует API **DeepGEMM 0.2.0**, закреплённый в базе 0.5.20; первоначальная
спецификация ссылалась на старый 0.1.7. Проверены Python API wheel и
[allocation layout release/v0.2.0](https://github.com/sgl-project/DeepGEMM/blob/release/v0.2.0/csrc/apis/attention.hpp):
FP32 stride кратен 256 колонкам, legacy имеет дополнительный block KV,
compact — максимальную длину запроса. Бюджет 1024 MiB относится к **logits**;
KV staging, top-k, activations и CUDA graphs требуют дополнительной памяти.
`SGLANG_DSA_FUSE_TOPK=0` сохранён. Для одного запроса compact не сужает контекст.

Особенность P4: текущий SM90 Q8 kernel возвращает **natural-log LSE**, хотя
считает softmax через exp2; его empty sentinel — `+inf`. Адаптер явно переводит
LSE в log2 и возвращает `−inf`/нулевой output для пустого shard. Это закреплено
CPU-тестом и проверяется CUDA-паритетом. Владельцы определяются по значениям
allocator read table через KVIndexTranslator; физическая трансляция сохранена.
CP-gather **новых** KV и запись persistent KV остаются прежними; убран полный
**prefix** KV all-gather. Q-stream пока ограничен GLM W4AFP8, SM90,
CP8/interleave/DCP4/EP8/DP1/PP1, top-k 2048 и MLA 512+64.

P6 оставляет indexer, динамические metadata и подготовку локального KV вне
фиксированного tile graph. Буферы KV округлены до степени двойки. Смена их
адресов/ёмкости инвалидирует граф и вызывает отдельный JIT/capture этап.
Первичные формы прогреваются до readiness; новый, больший контекст может
потребовать повторного capture. Его время пишется отдельно и не должно
попадать в steady-state сравнение. Весь attention одним графом не захватывается.

P5 не разрешает подключить storage/linker во время работы pipelined режима.
Существующие producer/write fences и синхронизация finish events сохранены.
Дайджест проверяется относительно отправленного snapshot, не изменившегося дерева.

## Профили и сравнение

`profile.json` — P2-кандидат; `profile-control.json` — P1-контроль. Каталог
`profiles/` содержит последовательные варианты с `comparison_parent`:

1. `p1-control-16k` → `p2-adaptive-16k`: scheduler.
2. `p2-adaptive-16k` → `p3-legacy-1g`: только явный workspace и форма порций.
3. `p3-legacy-1g` → `p3-compact-1g`: только compact layout.
4. `p3-compact-1g` → `p4-q-stream`: распределённый attention.
5. `p4-q-stream` → `p5-pipelined`: HiCache consensus.
6. `p5-pipelined` → `p6-buckets` → `p6-attention-graph`: buckets и tile capture раздельно.

Во всех вариантах сохранены текущие target/draft weights, Humming, полный
DFlash-кэш, MoE/DeepEP настройки и colocated serving. Decode fusion/bounded
cache/смена MoE backend в серию не включены.

Из установленного bundle запускать, например:

```bash
python3 launch_profile.py profiles/p2-adaptive-16k.json --dry-run
# Реальный запуск — отдельный следующий этап:
python3 launch_profile.py profiles/p2-adaptive-16k.json --kv-tokens SAME_KV_CAP
```

`SAME_KV_CAP` нужно заменить общим числом **эффективных** KV-токенов из
`glm53_effective_runtime` на всех сравниваемых вариантах. Ограничение передаётся
как `--max-total-tokens`; если runtime по памяти выдаёт меньше, уменьшить общий
cap для всех вариантов. `0.78` — старт, не гарантия. После OOM или остатка менее
2 GiB хотя бы на одной карте снижать через `--mem-fraction 0.76`, затем 0.74 и т.д.
Проверять пик **при работе**, а не только после старта; сохранять DCGM/NVML
выборки со всех восьми GPU и CUDA allocator peak из профиля. Одинаковая mem fraction
сама по себе не обеспечивает одинаковую KV-ёмкость.

## Проверки

CPU (из checkout; отдельное окружение с torch, pytest, msgspec, starlette, anyio):

```bash
python3 -m pytest -q deploy/glm53-v0.5.20/tests \
  --ignore=deploy/glm53-v0.5.20/tests/cuda
```

CUDA — выполнены на H200 в закреплённом образе с проверяемым source overlay; команды для повторения:

```bash
python3 -m pytest -q deploy/glm53-v0.5.20/tests/cuda/test_compact_logits.py
# Все четыре rank должны исполнять одинаковый набор pytest tests.
torchrun --standalone --nproc-per-node=4 -m pytest -q \
  deploy/glm53-v0.5.20/tests/cuda/test_qstream_distributed.py
```

Для bundle путь к тестам начинается с `tests/`. CUDA-тест проверяет legacy/compact
logits и top-k; distributed тест — пустой shard, `−1`, remapped IDs, Triton/CPU
compaction, native Q8 vs распределённый результат и FP32 reference, eager/graph.
Допуски native Q8 output сохранены `atol=rtol=0.08`, compact logits — точное
совпадение. CPU math reference имеет `1e-5/1e-6`. Это не заменяет GLM/DFlash
end-to-end parity и HiCache loadback: их нужно прогнать на полном сервере.

Клиент для отдельного тестового сервера (не запускает модель/не меняет кэши):

```bash
python3 validate_prefill_workload.py --base-url http://127.0.0.1:8080 \
  --scenario mixed --output /tmp/p2-mixed-run1
```

Сценарии: `fresh` (128k), `mixed` (128k, затем 4/5/6/7/10/15k), `constant`
(поток коротких), `burst` (40 независимых), `repeated` (две волны общего 32k
prefix). Входные token IDs и расписание детерминированы seed; диапазон IDs
100…29999 предназначен для текущего GLM vocabulary. Raw `/metrics` и данные
клиента сохраняются. Ключ берётся только из env `GLM53_TEST_API_KEY`; в файлы не
пишется. Cache flush не выполняется автоматически. Для fresh-сравнений обеспечить
одинаково пустой cache, для repeated — одинаково прогретый. Не использовать
synthetic inputs как проверку качества модели.

Сохранять фактические времена отправки: при исчерпании 41 клиентского worker
расписание может задержаться. SSE updates могут содержать несколько токенов;
их интервалы не являются чистым GPU TPOT. Queue time брать из scheduler telemetry,
TTFT — из клиента, chunk/kernel time — из P0/Nsight. Для детального Nsight-разбора
запускать сервер с диагностикой и отдельно фиксировать gather, conversion,
indexer/top-k, attention, MoE, draft append, consensus. CPU timings показывают
постановку GPU-работы, CUDA events — её stream-time; вложенные интервалы не складывать.

Prometheus-метрики P0 экспортирует только TP0/PP0; **их не делить на 8**.
RID присутствует только в JSON-журнале. Старые метрики с репликацией по rank
дедуплицировать отдельно. Основные метрики:
`sglang:glm53_prefill_admission_rejected_total`,
`sglang:glm53_prefill_stage_seconds`, `sglang:glm53_prefill_batch_tokens`.

Критерий scheduler: fitting-запросы проходят между чанками, долг компенсируется,
длинный не голодает, нет второго partial. Обязательно проверить нехватку KV,
host hit/miss/loadback, отмену и последний чанк. Критерий ускорения: несколько
сопоставимых повторений, улучшение своего участка около 5% или больше без
ухудшения общего latency/TPOT/корректности. Подтверждённые режимы переносить в
итоговый профиль только после этих проверок; текущая реализация не утверждает,
что q-stream быстрее kv-gather.
