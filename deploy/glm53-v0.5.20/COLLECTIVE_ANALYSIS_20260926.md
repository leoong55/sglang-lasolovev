# NCCL и ожидание других ranks — 26 сентября 2026

## Что измерено

В KV-gather профиле двух prefill-шагов TP0 сумма NCCL kernels составляет
475 ms при span всех kernels 1364 ms. Эти числа не доказывают, что транспорт
занимает треть времени: NCCL kernel может ждать поздно пришедших участников.

Проверены 150 ReduceScatter kernels на каждом из восьми ranks. Во всех trace
одинаковая Kineto timebase; сопоставление выполнено по порядку этих kernels.
Медианы: разброс времени входа 1.860 ms, выхода 0.117 ms; минимальная длительность
по ranks 0.535 ms, максимальная 2.334 ms. Это поддерживает гипотезу ожидания
поздних ranks внутри collective. Это диагностическая корреляция двух шагов,
не полное разложение TTFT и не доказательство постоянного bottleneck.

Два последних Humming kernels перед ReduceScatter занимают у позднего rank
медиану 2.895 ms, у раннего 1.103 ms. Поздний rank меняется между слоями,
чаще всего это rank6 (37/150), затем rank0 (26/150), rank3 (22/150).
Одного постоянно медленного GPU нет. Вероятная причина — различная работа
экспертов MoE, но без token counts нельзя отделить routing imbalance от
других различий запуска/форм kernel и влияния профайлера.

## Ограниченная проверка NVLS

Тот же образ, восемь H200, NCCL2.30.7, NV18 между всеми GPU. PyNccl проверен
с BF16 tensors [4096/8192/16384,6144], ReduceScatter и AllGather,
eager и CUDA graph, порядок off/on/on/off. Пять измерительных групп по30
операций на каждую форму; между группами barrier. Все численные проверки
на константных rank-зависимых tensors прошли на восьми ranks.

Для16k graph median по16 rank/run observations:

| Операция | NVLS off, ms | NVLS on, ms |
|---|---:|---:|
| ReduceScatter |0.5598|0.5578|
| AllGather |0.5654|0.5632|

Практического выигрыша нет. Логи on подтверждают наличие NVLS multicast,
но проверенные коллективы выбирают **Ring/Simple**. Поэтому нельзя говорить,
что сам NVLS-алгоритм медленный: он здесь не выбран. Полный GLM-прогон с одним
этим флагом не оправдан, флаг остаётся выключенным.

NCCL использует модель стоимости для выбора алгоритмов; регистрация пользовательских
буферов — отдельный механизм, требующий корректного allocator и согласованности
всех ranks. В этой проверке allocator не менялся. Название RING_LL в trace
не доказывает LL-протокол; TUNING log в probe явно показывает Simple.

Первичные источники:

- [NCCL2.30.7 cost model](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/graph/tuning.cc).
- [NVIDIA: buffer registration, allocator и согласованность ranks](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2303/user-guide/docs/usage/bufferreg.html).
- [NCCL issue о неоднозначных названиях RING_LL](https://github.com/NVIDIA/nccl/issues/2196).
- [SGLang Expert Parallelism и EPLB](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx).

## Следующий различающий опыт

Вариант p7-expertstats-r1 сохраняет P6buckets и включает только штатный stat
recorder, без EPLB или перемещения весов. После exact nonce/natural-stop smoke
три новых calibration inputs128k/1out позволят снять распределение tokens
по экспертам/слоям. Эти запросы не входят в performance A/B. Калибровка
синтетическая, вывод о производственных текстах потребует отдельной проверки.

Сначала подтвердить или отвергнуть imbalance. Только затем выбирать балансировку
или адресное изменение Humming; не подменять ожидание GPU «медленным NCCL».


## Expert counts подтверждают источник ожидания

P7 прошёл exact nonce/natural stop и три отдельных calibration128k/1out.
В native recorder сохранены24полныхprefillшага,78слоёв,256экспертов;75слоёв MoE.
На текущем contiguous размещении max/mean работы восьми EP ranks:
median1.701, p95=2.555. Это отношение routing counts, не SM utilization.
Сумма счётчиков включает одинаковые наблюдения ranks; абсолютное число
не выдаётся за число уникальных входных токенов.

Для138из150collectives предыдущего trace поздний rank совпадает с самым
нагруженным rank по counts;149/150попадают в два самых нагруженных. Входы
trace и калибровки различаются; сопоставление исходит из порядка75MoEслоёв
дважды. Это сильное независимое подтверждение MoE-skew в данной синтетической
нагрузке, а не доказательство такого же распределения на произвольном проде.

Подготовлен один адресный опыт P8: фиксированная перестановка256экспертов
на каждомслое,32/GPU, через штатный --init-expert-location и rank-invariant
--ep-dispatch-algorithm dynamic. Без дубликатов, онлайн EPLB, изменения
весов, Humming, CP/DCP/TP/EP, KV cap или cache policy.
Алгоритм — capacity-constrained greedy packing, как в
[DeepSeek EPLB](https://github.com/deepseek-ai/EPLB/blob/main/eplb.py);
[штатный SGLang init expert location](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/eplb/expert_location.py)
читает physical_to_logical_map. Локальная закреплённая версия проверена по коду
loader/remap; draft-worker не перезаписывает target metadata.

Карта построена по первым двум новым calibration inputs; третий отложен.
На третьем прогноз max/mean median1.701→1.079, p95=2.577→1.215;
сумма наибольшей нагрузки по слоям/шагам меньше39.0%.
**Это расчёт token load, не измеренное ускорение модели.**
P8 ещё не запущен; обязательны численная проверка и сопоставимый workload.
См.expert-count-analysis.json, expert-trace-correlation.json и
p8-expertlayout-r1-design.json.
