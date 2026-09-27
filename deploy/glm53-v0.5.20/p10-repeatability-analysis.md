# P10: границы результата проверки повторяемости

Stream-K-off устранил расхождения в компонентном MoE опыте (64/64 точек), но не в полном GLM runtime. В matched-cache P10 завершены все29запросов; первый выходной token ID совпал во всех сравнениях, но32-token последовательности и первые top20 logprobs не совпали точно.

| Сравнение с resident-before-0 | P9 max shared logprob diff | P10 |
|---|---:|---:|
| resident-before-1 |0.4882|0.5047|
| resident-before-2 |0.2852|0.2599|
| host reload |0.1973|0.2969|

Это не доказательство улучшения или ухудшения точности. У нас нет точных полных reference logits, а пересечение top20 не измеряет всю ошибку распределения. Расхождения resident-before возникают без host reload, поэтому этот тест не локализует дефект HiCache.

Проверена гипотеза о примеси health-запроса или другой форме scheduler batch: во всех семи сравниваемых P10 prefills TP0 логирует ровно1запрос,256новыхтокенов и130816токеновprefix. См.`p10-cache-batch-shapes.json`. Внешний batch одинаковый; внутренние kernel plans, адреса KV, reduction order и происхождение возвращённых logprobs этим не проверены. Объяснять результат только разным размером batch нельзя.

По исходникам W4AFP8 create_moe_runner передаёт `_humming_disable_stream_k` в слой; get_humming_gemm_configs применяет `use_stream_k=False` к w13 иw2 до сохранения конфигурации, для indexed/grouped путей. Это локальная правка двух expert GEMM. Она не меняет FP8 dense projections, DSA/indexer, attention, распределённое суммирование и DFlash. Следовательно компонентная повторяемость не гарантирует повторяемость полного пути. Остаточная причина пока не локализована; новый runtime патч без такого основания не добавляем.

Upstream прямо описывает влияние порядка floating-point reduction на результаты даже при temperature0. В [документации deterministic inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference) перечислены FlashInfer/FA3/Triton; наш FlashMLA sparse Q8 + DCP + Humming + DFlash не является этим задокументированным набором гарантий. Это контекст, а не доказательство причины наших расхождений. Глобальный deterministic flag не включался: это изменило бы другие механизмы и условия сравнения.

Никакие допуски не ослаблены. Пока сохраняется статус full-model/cache numerical gate inconclusive. Профиль GPU P10 завершён отдельно после всех performance/cache запросов; восемь trace и архив проверены SHA256. Анализ времени сохранён в p10-streamk-analysis.json; сам по себе он не является проверкой точности.
