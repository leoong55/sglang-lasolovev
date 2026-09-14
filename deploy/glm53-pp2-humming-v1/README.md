# GLM-5.3: TP4 × PP2, EP4, Humming, CUDA-графы

Кандидат для восьми H200 на основе `work/glm53-stg-pp2-dpa` / PR #11,
commit `eeec88d3467de518d46b27680ce70d4361a4f422`. Upstream:
`0bcd822377da7b5718e674eaf9c870d349424dd1`.

Из v9.10 `1f1171df23c98755c7d479d8befe071a9e13ecbe` перенесены ровно три
runtime-файла: выбор Humming для W4AFP8 MoE, адаптер загрузки этих весов и
поправка оценки размера локальной работы EP. Идентичность файлов фиксирует
`manifest.json`. Изменений PP-планировщика, внимания, GPU KV и CUDA-графов нет.
Старый STG launcher и его протокол измерений сохранены в отдельном каталоге.

Модель: полная `PhalaCloud/GLM-5.3-W4AFP8`, 78 слоёв; разделение PP — 39/39.
На каждом этапе TP4/EP4, итого восемь GPU. DCP1, CP выключен, DP1.
HiCache и спекулятивка выключены; обычный GPU radix cache включён.

## Графы и параметры

| Параметр | Значение |
| --- | --- |
| Prefill | `flashmla_kv`, chunk 16384 |
| Prefill CUDA-графы | `breakable`, 512/1024/2048/4096/8192/16384 токенов |
| Decode | `flashmla_kv`, `full` CUDA-графы |
| Decode CUDA-графы | 1/2/4/8/12/16/20/24 запросов на микробатч |
| Запросы / микробатч | 48 / 24; PP async depth 0 |
| KV | FP8 E4M3, page 64 |
| Контекст | 98304, как в PP2-заготовке; это явный изменяемый лимит |
| GPU-память | `mem-fraction-static=0.80`, как в PP2-заготовке |
| Humming | 0.1.12, indexed, FP32 accumulation (`USE_F16_ACCUM=0`) |

В отличие от CP8, здесь чанк 16384 не делится по CP: каждый PP-этап обрабатывает
весь чанк своими четырьмя GPU. Prefill-графы PP выключены по умолчанию в upstream,
но поддерживают явное включение `breakable`: `arg_groups/cuda_graph_hook.py`,
`model_executor/model_runner_components/cuda_graph_setup.py` и
`model_executor/runner/prefill_cuda_graph_runner.py`. Последний передаёт и
обрезает PP proxy tensors, включая residual и переиспользуемые индексы DSA.
Комбинация PP с prefill CP остаётся выключенной. `tc_piecewise` для PP не выбран.

Prefill-графы могут не ускорить крупные чанки. Для проверки без пересборки можно
сменить только `--cuda-graph-backend-prefill` на `disabled`. Decode-графы при этом
останутся включены. Все штатные аргументы проходят через `launch.py` без подмены;
новых обязательных проверок фактического replay в launcher нет.

Число 24 в decode-графах соответствует `--pp-max-micro-batch-size`, а не общему
лимиту 48. При увеличении микробатча нужно также расширить decode-графы. Native
PP отключает overlap scheduler; конвейер PP работает через собственный цикл.

`SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.05` ограничивает временный бюджет DSA,
а не размер всего KV. Сохранено отключение FlashInfer allreduce fusion, чтобы
этот вариант не зависел от дополнительного пути fusion из CP-эксперимента.
Если при захвате графов не хватает GPU RAM, доступны обычные ручки памяти,
размера чанка и prefill-графов; автоматического уменьшения в launcher нет.

## Спекулятивка с PP2

В закреплённом runtime она не поддерживается. Это ограничение upstream:

- `arg_groups/speculative_hook.py::_handle_dflash` требует `pp_size == 1`;
- `arg_groups/validation_hook.py::check_server_args` запрещает любую
  спекулятивку с PP>1, включая EAGLE/MTP;
- speculative workers создают draft без pipeline parallelism.

Удалять эти проверки недостаточно. Понадобятся согласованная между PP-этапами
передача draft/verify состояний, подтверждённых токенов и освобождения KV,
работа с микробатчами и отдельная проверка корректности. Humming сам по себе
этого ограничения не снимает. Невозможность относится к текущей реализации,
а не к принципиальной возможности спекулятивки с pipeline parallelism.

## Сборка

Сборка использует закреплённый публичный SGLang CUDA 13 образ с уже установленным
`humming-kernels==0.1.12`. Она копирует три runtime-файла и проверяет их, выполняет
CPU-тесты загрузки/упаковки Humming. Нет установки новой версии Torch, сборки
CUTLASS или nvcc-компиляции. JIT под реальные GPU произойдёт при запуске модели.

Из корня чистого checkout этой ветки:

```bash
PP2_COMMIT=$(git rev-parse HEAD)
PP2_IMAGE=i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-pp2-humming-v1-${PP2_COMMIT:0:12}

docker build --progress=plain \
  --build-arg SOURCE_COMMIT="$PP2_COMMIT" \
  -f deploy/glm53-pp2-humming-v1/Dockerfile \
  -t "$PP2_IMAGE" .

docker push "$PP2_IMAGE"
```

Образ CP/DCP v9.10–v9.13 не является базой для этого overlay. Если используется
зеркало базового образа в Harbor, задайте его через `--build-arg BASE_IMAGE=...`;
проверка исходников и зависимости должна пройти. Dockerfile-specific ignore
включает только каталог сборки и три runtime-файла: веса и переписки в контекст
сборки не попадают.

GitHub workflow `GLM53 PP2 Humming image` также собирает образ и публикует тег
`ghcr.io/leoong55/sglang-lasolovev:glm53-pp2-humming-v1-<commit12>`.
Успешный run отдаёт `image.json` и манифест с digest. Это подтверждает сборку и
CPU-проверки, но не GPU-запуск или производительность.

## Запуск на текущем стенде

```bash
python3 deploy/glm53-pp2-humming-v1/render.py \
  --image "$PP2_IMAGE" > 06-pp2-humming.yaml
kubectl -n inf-glm53 apply -f 06-pp2-humming.yaml
kubectl -n inf-glm53 rollout status deployment/sglang-glm53-dcp4 --timeout=7200s
```

Манифест заменяет существующий `sglang-glm53-dcp4` через `Recreate`; имени
Service достаточно, чтобы сохранить текущий адрес бенчмарка. Новая копия модели
и дополнительные GPU не нужны. Используются существующие model/compile-cache
PVC, прежние ресурсы 112 CPU / 1536 GiB / восемь GPU и стандартный scheduler
из присланного PROD-манифеста. Это не HAMi-шаблон из STG.

Readiness проверяет `/model_info`: в прежнем PP-тесте генерирующий `/health`
не укладывался в timeout во время длинного prefill. Startup по-прежнему проверяет
`/health`. API readiness не является проверкой качества генерации.

Для сверки версии и запуска:

```bash
kubectl -n inf-glm53 exec deployment/sglang-glm53-dcp4 -c sglang -- \
  cat /opt/glm53-pp2-humming-v1/REVISION.json
kubectl -n inf-glm53 logs deployment/sglang-glm53-dcp4 -c sglang | \
  grep -E 'GLM53_PP2_IMAGE|W4AFP8 MoE: Humming|Captur|CUDA graph|max_total_num_tokens'
```

После старта нужны генерация и GSM8K для новой TP4/PP2 геометрии. Предыдущая
проверка точности TP8/CP8 не подтверждает автоматически эту комбинацию.
Дальше — те же короткий C40 и длинный 60k+15k→1k C40 бенчмарки.
Архив переписки описывает прежний PP2 как подготовительные прогоны, в том числе
с компиляцией: их 365/394 output tok/s нельзя выдавать за чистый A/B этого патча.
Ускорение и объём доступного KV новой сборки пока не измерены.
