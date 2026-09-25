# Реестр переноса на 0.5.20

Upstream: [v0.5.20](https://github.com/sgl-project/sglang/releases/tag/v0.5.20),
commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.
Наш предыдущий runtime: `219206f7db9f7ffb24dad2605edf72af62d4939e` поверх
`0bcd822377da7b5718e674eaf9c870d349424dd1`.

| Группа | Что сохранено / адаптировано |
|---|---|
| DSA + Hopper DCP | Backport #36990: Q gather/LSE reduction, packed FP8 prefix KV, индексы gathered CP-prefill и rank-local decode, causal verify rows. В новом upstream planner использует KVIndexTranslator: сохранён новый перевод widened→local, без повторного деления. |
| CP collectives | Переиспользование рабочих буферов межслойного gather; выход не alias-ит временный transport buffer. Нулевой prefix пропускает collectives, сохраняя upstream HIP fix. |
| BCG interleave CP8 | Eager attention break, fixed transformer rows, live DCP plan, глобальные DFlash hidden features, bounded padding до 25%, buckets 8k/16k/32k. API `is_cp_active` вместо удалённого `is_cp_v2_active`. |
| Short prefill | Сохранены constexpr capture geometry и выбор natural-log LSE по фактически выбранному backend, включая короткий prefill с fallback Q8→FlashMLA-KV. |
| W4AFP8 + Humming | Repack исходных signed INT4, persistent MoE runner, EP-aware padding и прежний контроль Humming 0.1.12. Другие upstream quantization backends не заменяются. |
| DFlash CP8/DCP4 | Full-TP draft/head geometry, causal mask, scheduler graph vote и MLP metadata, защита sampler buffers при batch выше capture capacity. Новый upstream DP-local draft context отключается только для нашего узкого CP8/DCP4-контракта, сохраняя TP8 draft и его graphs. |
| HiCache | Полный DFlash sidecar, producer/write fences, target/indexer host mapping, replicated DSA index pages и sparse index elision. Сохранён upstream запрет elision с unified external linker. |
| Необязательные режимы | Bounded GPU draft 2048, short verify block 2/4/8, diagnostics `warn/require`, timings, CP decode attention fusion. Код сохранён; текущий профиль их не включает. |
| Отмена/SSE | Upstream #35255 уже содержит dispatched/abort_sent, async dispatch и batch waiter cleanup. Старые дублирующие state fields не перенесены. Добавлены identity guard для reused RID, nested aclosing и shielded SSE cleanup; сохранены upstream encoder dispatch events и retry abort. |
| Reasoning | Перенесён C1 commit `706379ac116d8bc048fe080597ba486f86864d35`: явный request effort выше server default, явные template kwargs выше defaults, medium→high как в reasoning-fix4. Это сохранённая политика нашего GLM-образа. |
| C1 меньший чанк | Источник `a55c416d1:deploy/glm53-c1/launch.py`: 4096 разрешён только с prefill graphs disabled. Перенесено в обычный launcher, без подмены SUPPORTED_CHUNKS во время запуска. Сохранены native TP8/DCP1 comparison routes. |

Из upstream оставлены новые sink/kpool проверки DSA, несpecialизированный runtime
stride kernel page table, новые graph variant API, XPU/DP DFlash paths,
защиты deferred scheduler abort и актуальная схема конфигурации. Удалённые CP-v1
поля и runtime imports не восстанавливались. Изменения старой release-ветки вне
нашего списка файлов не переопределяют содержимое 0.5.20.

История накопленной серии: [patch-history.txt](patch-history.txt). Список runtime
файлов с SHA256 старой версии, upstream и перенесённого содержимого:
[patch-inventory.json](patch-inventory.json). Генерируемый `runtime.patch` содержит
ровно diff этого дерева относительно указанного upstream commit.

Статус: разрешены конфликты и выполнены статические проверки синтаксиса/ссылок;
образ, импорт полного движка, host/GPU-тесты и численная корректность не проверялись.
