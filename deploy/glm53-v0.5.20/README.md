# GLM-5.3 W4AFP8 на SGLang 0.5.20

Исходники подготовлены для отдельной сборки и проверки на H200. Образ в рамках
этого переноса не собирался; host-, kernel- и GPU-тесты не запускались.

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
DCP translation, draft worker и decode graph runner. Несовпадение останавливает
установку до изменения файлов.

Экспорт содержит `runtime.patch`, overlay, `base-files.json`, `REVISION.json`,
`SHA256SUMS`, Dockerfile и launcher. Все runtime-изменения хранятся обычным кодом
в `python/sglang`, а patch/overlay генерируются из зафиксированного Git tree.
Скрипт не публикует образ. Не запускайте Dockerfile напрямую из каталога kit:
ему нужны overlay и checksums, созданные `make_bundle.py`.

## Профиль запуска

[profile.json](profile.json) сохраняет argv/env присланного манифеста: TP8/EP8,
CP8 interleave, DCP4 ag_rs, Humming, FP8 KV, HiCache L1/L2 96 GiB, чанк 8192,
DFlash2 block 8, **полный draft-кэш** (`window=0`), graph policy `warn`.
`weight-version` заменён на `glm53-v0.5.20-candidate`; удалён устаревший
`SGLANG_ENABLE_CP_V2`. Параметры графов, включая decode buckets 1/2/4/6/8/12/16/
20/24/32/40, сохранены. HF checkpoint оставлен как в присланном манифесте.

Новый entrypoint: `/opt/glm53-v0.5.20/launch.py`. Совместимый симлинк
`/opt/glm53-cp8-dcp4-v1` сохраняет старую команду манифеста. Каталог модели и
кэшей по-прежнему передаётся через mounts и env; kit не меняет PVC или кластер.

Из C1 перенесён чанк **4096 только с prefill graphs disabled**. Capture buckets
BCG остаются 8192/16384/32768. Сохранены необязательные bounded draft 2048,
CP decode fusion и native TP8 comparisons (DFLASH, DSPARK, EAGLE/MTP); они не
включены в текущем профиле. Для native TP8 требуются DCP1 и отсутствие CP-флагов.
Bounded draft остаётся ограничен CP8/DCP4/DFLASH.

## Проверка позже

`tests/` содержит перенесённые регрессии SSE/отмены с новым async dispatch API и
сценарии launcher 4k/8k. Они подготовлены, **не запущены**. Для будущего CPU-прогона:

```bash
python3 -m pip install starlette==0.48.0 anyio==4.10.0
SGLANG_SOURCE_ROOT="$PWD" python3 -m unittest discover \
  -s deploy/glm53-v0.5.20/tests -v
```

Старые kernel/host-сценарии находятся в исторических `deploy/glm53-hicache-v9`
и `test/registered/dcp`. Их ожидания относятся к прежнему runtime; часть fixture
нуждается в адаптации к удалённому CP-v1 API. Не считать старый зелёный результат
проверкой 0.5.20. `kernel_preflight.py` сохранён как ручной инструмент компиляции.

Перед эксплуатацией нужны сборка, загрузка текущих весов, корректная генерация,
проверка draft/verify/sampler graphs, HiCache round-trip и отмены, затем отдельные
замеры. Изменились upstream-зависимости (`sglang-kernel 0.4.7`,
`sgl-deep-gemm 0.2.0`); перенос исходников сам по себе не подтверждает скорость
или корректность на GPU.

Каталоги `glm53-*-v1…v9`, их Dockerfiles, manifests и отчёты сохранены как история.
Актуальная точка сборки этой ветки — **этот каталог**.
