# GLM-5.3 W4AFP8 на SGLang 0.5.20

Серия P0–P6 реализована поверх подготовленной базы 0.5.20; затем добавлены
адресные исправления P8–P10. Кампания на 8 H200 завершила 5643 performance-запроса
в девяти вариантах. Итоги и границы корректности: [H200_FINAL_20260927.md](H200_FINAL_20260927.md).
Точные измеренные конфигурации: [measured-20260927/](measured-20260927/).
Универсальный production-профиль не принят: full-model numerical/cache gate
остаётся inconclusive, абсолютный пик памяти не измерен. Образ не собирался.
Актуальные переключатели, этапы сравнения и ограничения:
[PREFILL16K.md](PREFILL16K.md).

Основа: upstream `v0.5.20`, commit
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`. Накопленная серия нашего форка
сохранена merge-историей до `219206f7db9f7ffb24dad2605edf72af62d4939e` (v9.14).
Реестр изменений и решений при переносе: [PATCHES.md](PATCHES.md).
Все параметры базы и происхождение C1-изменений: [provenance.json](provenance.json).

## Будущая сборка

Запускать отдельно, когда будет принято решение собирать образ:

```bash
# Из чистого checkout этой ветки; выполняется только экспорт исходников.
python3 deploy/glm53-v0.5.20/make_bundle.py /tmp/glm53-v0.5.20-source

# Следующая команда уже собирает образ. В рамках переноса не запускалась.
cd /tmp/glm53-v0.5.20-source
./build.sh YOUR_REGISTRY/sglang:glm53-v0.5.20-candidate
```

Dockerfile принимает `BASE_IMAGE`; `build.sh` передаёт его через `GLM53_BASE_IMAGE`.
По умолчанию закреплён официальный `lmsysorg/sglang:v0.5.20`:
`sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f`.
Digest получен из Docker Registry 25.09.2026; слои не скачивались. Для зеркала
Harbor можно указать ссылку на копию **того же** образа. Старый v9.x образ для
этого installer не подходит: принимаются SHA256 файлов точной базы 0.5.20 либо
уже установленного overlay. Проверяются также неизменённые границы API CP,
draft worker и decode graph runner. Несовпадение останавливает
установку до изменения файлов.

Экспорт содержит `runtime.patch`, overlay, `base-files.json`, `REVISION.json`,
`SHA256SUMS`, Dockerfile и launcher. Все runtime-изменения хранятся обычным кодом
в `python/sglang`, а patch/overlay генерируются из зафиксированного Git tree.
Скрипт не публикует образ. Не запускайте Dockerfile напрямую из каталога kit:
ему нужны overlay и checksums, созданные `make_bundle.py`.

## Профиль запуска

[profile.json](profile.json) — кандидат 16k с adaptive interleaving и HRRN.
Сохранены TP8/CP8/DCP4/EP8, Humming, FP8 KV, HiCache L1/L2, текущие веса
и полный DFlash-кэш (`window=0`). `min_free_slots_delay=1` задан явно.
[profile-control.json](profile-control.json) — FIFO-контроль с теми же 16k.
P3–P6 представлены ранними cumulative-примерами в [profiles/](profiles/).
Для повторения измерений использовать [measured-20260927/](measured-20260927/):
там реальные изолированные сравнения, HiCache64 и одинаковый KV cap.
Ни один GPU-вариант не обозначен как полностью принятый. `mem_fraction_static=0.78` требует
проверки пика памяти с запасом 2 GiB на каждой карте.

Новый entrypoint: `/opt/glm53-v0.5.20/launch.py`. Совместимый симлинк
`/opt/glm53-cp8-dcp4-v1` сохраняет старую команду манифеста. Каталог модели и
кэшей по-прежнему передаётся через mounts и env; kit не меняет PVC или кластер.

Capture buckets BCG: 4096/8192/12288/16384; прежний 32768 сохранён как
отдельная доступная опция. В серии общий бюджет остаётся 16384. Сохранены необязательные bounded draft 2048,
CP decode fusion и native TP8 comparisons (DFLASH, DSPARK, EAGLE/MTP); они не
включены в текущем профиле. Для native TP8 требуются DCP1 и отсутствие CP-флагов.
Bounded draft остаётся ограничен CP8/DCP4/DFLASH.

## Проверки и оставшиеся ограничения

`tests/` содержит CPU-регрессии admission, компенсации бюджета, logits,
LSE, HiCache snapshots, профилей, CLI и отмены/SSE. Команды воспроизведения
и CUDA-проверки приведены в [PREFILL16K.md](PREFILL16K.md).

Старые kernel/host-сценарии находятся в исторических `deploy/glm53-hicache-v9`
и `test/registered/dcp`. Их ожидания относятся к прежнему runtime; часть fixture
нуждается в адаптации к удалённому CP-v1 API. Не считать старый зелёный результат
проверкой 0.5.20. `kernel_preflight.py` сохранён как ручной инструмент компиляции.

Загрузка текущих весов, nonce/natural-stop генерация, CUDA capture и performance
проверены в кампании; компонентные проверки отражены в [validation-status.json](validation-status.json).
Перед эксплуатацией остаются сборка, полная численная проверка, проверка памяти
и production-нагрузки. Побайтовый DMA gate не равен полной корректности HiCache.
Изменились upstream-зависимости (`sglang-kernel 0.4.7`,
`sgl-deep-gemm 0.2.0`); перенос исходников сам по себе не подтверждает скорость
или корректность на GPU.

Каталоги `glm53-*-v1…v9`, их Dockerfiles, manifests и отчёты сохранены как история.
Актуальная точка сборки этой ветки — **этот каталог**.
