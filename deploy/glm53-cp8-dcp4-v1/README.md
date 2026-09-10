# GLM-5.3 CP8 + DCP4: первый запуск на H200

Для специализированного FP8 prefill подготовлен [комплект q8-v2](../glm53-cp8-dcp4-q8-v2/README.md).
Он требует нового образа; GPU-проверка ещё предстоит. Исходный v1 зафиксирован
коммитом `7f7331fa4408ea32b1e508020b6a860a8cfea66d`.

Ручной backport Hopper DSA DCP из [SGLang PR #36990](https://github.com/sgl-project/sglang/pull/36990),
head `c6aeb8b9d9128b816777e2b64cbf6603344e8fe1`, на базу
`0bcd822377da7b5718e674eaf9c870d349424dd1`. Исходная работа — автор PR и SGLang contributors,
Apache-2.0. Сохранены изменения нашей базы, включая `DSATopKBackend.resolve(model_runner)`
и аргумент gating в MLA forward. Проверка CP/DCP перенесена в актуальный
`arg_groups/parallel_hook.py`; дополнительно запрещён HiCache для DSA DCP.

Это первая сборка для проверки на железе. Локально прошли 9 CPU-тестов host-логики,
проверка синтаксиса, наложения патча и установщика. CUDA, FlashMLA, NCCL и модель на
8 GPU здесь не запускались. Docker-образ в этом окружении не собран: Docker/BuildKit
отсутствуют. Архив содержит исходный патч и готовый контекст сборки, а не готовый образ.

## Профиль

После первого запуска: [контрольный запуск с decode CUDA graphs (A1)](diagnostics/README.md).
Он подготовлен по профилю медленного eager decode; эффективность требует проверки
на GPU. Исходный манифест ниже остаётся воспроизводимым bring-up baseline.

| Параметр | Значение |
|---|---|
| Модель | Полная `PhalaCloud/GLM-5.3-W4AFP8`, 78 слоёв; не Flash |
| GPU / ресурсы pod | 8×H200 / 112 CPU / 1536 GiB RAM |
| Параллелизм | TP8, EP8, prefill CP8 interleave, decode DCP4, DP1 |
| Decode attention | `--enable-cp-decode-attn-tp` |
| Prefill / decode backend | `flashmla_kv` / `flashmla_kv` |
| Квантизация | W4AFP8 весов, `fp8_e4m3` KV |
| Page / chunk | 64 / 8192 |
| Static memory / requests | 0.80 / 32 |
| HiCache / speculative / CUDA graphs | Выключены |

DP4 вместе с CP8 на этих восьми GPU не подходит для выбранного interleave-пути:
проверка требует `tp_size % (dp_size * attn_cp_size) == 0`.
DCP4 использует группы внутри тех же восьми GPU; 32 GPU не нужны. CP8 получается
из TP8/DP1 и interleave, отдельного пользовательского `--cp-size 8` здесь нет.
MoE runner оставлен штатному выбору W4AFP8 в закреплённой базе.

Основной MLA KV распределяется по DCP-рангам; index-K остаётся реплицированным.
Учёт памяти indexer исправлен с учётом DCP. Поэтому рост ёмкости не равен строго 4×.
Количество токенов определяет runtime по доступной памяти; лимит `max-total-tokens`
не задан. Значения локального KV-пула и логической ёмкости DCP могут различаться.
Не умножайте любое число из лога на 4 без проверки, что именно оно обозначает.

При CP+DCP prefill этот PR собирает KV префикса в временный буфер. Длинная история
увеличивает обмены и пиковую память. Chunk 8192 ограничивает новые токены prefill,
но не весь собранный префикс. Начальная проверка не доказывает работу на миллионах
токенов; сначала нужны короткие запросы, затем chunked prefill и повторный префикс.

## Сборка

Из checkout с этим коммитом сначала сформировать комплект вне репозитория:

```bash
python3 deploy/glm53-cp8-dcp4-v1/make_bundle.py /tmp/sglang-glm53-cp8-dcp4-v1
cd /tmp/sglang-glm53-cp8-dcp4-v1
```

В готовом release-архиве этот шаг уже выполнен. Распакуйте его и перейдите в каталог.

```bash
sha256sum -c SHA256SUMS
docker login i501-harbor-infra.ai.turbocloud.ru
bash build.sh
docker push i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-cp8-dcp4-v1-0bcd822377da
```

`build.sh` собирает производный образ, проверяет установленный overlay и создаёт
`sglang-glm53-cp8-dcp4-v1-image.tar.gz` через `docker save`, SHA-256 и `image-inspect.json`.
Для импорта: `gzip -dc sglang-glm53-cp8-dcp4-v1-image.tar.gz | docker load`.
Проверьте свободное место для большого базового образа и экспорта.

База по умолчанию:
`i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:v0.5.19-latest-0bcd822377da`.
Переменная `BASE_IMAGE` позволяет указать digest того же образа для воспроизводимости;
`TARGET_IMAGE` — другой выходной tag. При смене tag измените также `manifest.yaml`.
Зависимости и бинарные CUDA-библиотеки не обновляются; используются из базового образа.

Установщик определяет фактически импортируемый Python-пакет `sglang` и до записи
проверяет все исходные SHA-256 из `base-files.json`. Несовпадение останавливает сборку;
не обходите эту проверку — нужен diff содержимого конкретного образа. Повторная
установка того же патча допустима. При старте снова проверяются итоговые SHA-256.

## Kubernetes

`manifest.yaml` содержит Namespace, compile-cache PVC, Deployment и ClusterIP Service.
ModelRun, draft PVC, HiCache и дополнительные контроллеры не нужны.

Перед apply проверьте существующий model PVC: в манифесте сохранено имя из примера
`sglang-flash-pvc` в namespace `inf-glm53`. Он должен содержать **полную** модель
`PhalaCloud/GLM-5.3-W4AFP8`, с `config.json` и весами в корне. Если она лежит в другом
PVC, измените `volumes[name=model].persistentVolumeClaim.claimName`; если в подкаталоге,
добавьте правильный `subPath` в model mount. PVC из другого namespace подключить нельзя.
Этот комплект не скачивает веса и не создаёт модельный PVC автоматически.

Секрет `harbor-pull` должен существовать в `inf-glm53`. Разместите pod на узле с восемью
H200; при необходимости добавьте `nodeSelector`/tolerations по реальным меткам кластера.
Имена GPU-меток в вашем кластере неизвестны, поэтому выдуманный selector не добавлен.
Launcher дополнительно проверяет форму конфигурации полной модели, чтобы не запустить
Flash по ошибке. Форма конфигурации не заменяет проверку происхождения самих весов.

```bash
kubectl apply -f manifest.yaml
kubectl -n inf-glm53 logs -f deployment/sglang-glm53-dcp4 --all-containers=true
kubectl -n inf-glm53 rollout status deployment/sglang-glm53-dcp4 --timeout=120m
kubectl -n inf-glm53 port-forward service/sglang-glm53-dcp4 8080:8080
```

Первые две команды наблюдения выполняйте в отдельных терминалах. Service доступен
внутри кластера; авторизация ModelRun/extAuth и внешний Gateway в этот манифест не перенесены.

После readiness, через port-forward:

```bash
python3 smoke.py --url http://127.0.0.1:8080 --output smoke-results.json
```

Smoke делает короткий completion, prefill длиннее chunk 8192 и повторяет тот же
префикс. Он проверяет ошибки HTTP, непустую генерацию и число входных токенов,
но не численную эквивалентность DCP и контрольной конфигурации.
В логе сервер должен показывать DCP4 и CP8, FP8 KV и выбранные backend'ы.
Не начинайте с нагрузочного теста на 32 длинных запроса.

Если pod не стартует, сохраните полный startup log со всех рангов, `kubectl describe pod`
и конфигурацию GPU (`nvidia-smi`). Если ошибка только на запросе — также `smoke-results.json`
и traceback. При OOM в prefill отдельно нужны длина префикса и текущая конкуренция;
свободная ёмкость постоянного KV-пула не гарантирует место под gather-буфер.

## Проверки и применение исходного diff

CPU harness использует AST для загрузки **настоящих функций изменённого исходника**,
обходя импорты CUDA. Triton launch, NCCL и FP8 quantizer заменяются тестовыми объектами.
Проверены layout/padding буфера, byte-exact FP8 prefix/append, выбор запросов CP,
bucket голов FlashMLA и CPU reference LSE с пустыми шардами. Это не запуск GPU-ядер.

Из checkout (нужен CPU PyTorch):

```bash
python3 -m unittest discover -s deploy/glm53-cp8-dcp4-v1/tests -p 'test_host*.py' -v
```

Из архива:

```bash
SGLANG_SOURCE_ROOT="$PWD/overlay" python3 -m unittest discover -s tests -p 'test_host*.py' -v
```

Отдельный `tests/test_dsa_dcp_kv_gather_integration.py` в архиве — тесты автора PR
с реальными импортами SGLang. Они также используют mocks коллективов и quantizer;
локально этот полный import-путь не запускался. Тест размещён в checkout как
`test/registered/dcp/test_dsa_dcp_kv_gather.py`.

`runtime.patch` содержит только изменения runtime. На чистом checkout базы:

```bash
git apply --check /path/to/bundle/runtime.patch
git apply /path/to/bundle/runtime.patch
```

Не применяйте diff повторно поверх ветки с уже установленным backport.
