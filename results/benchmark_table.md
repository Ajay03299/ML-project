Best baseline: **Ridge** (test MAE 65.11). Positive `vs_best_baseline_%` = better than that baseline.

| model                | kind     | owner   |   test_MAE |   test_RMSE |   test_R2 |   test_WAPE |   val_MAE |   vs_best_baseline_% |   train_time_s |   predict_ms_per_1000 |   model_size_kb |
|:---------------------|:---------|:--------|-----------:|------------:|----------:|------------:|----------:|---------------------:|---------------:|----------------------:|----------------:|
| XGBoost              | ML       | Anirudh |     42.04  |      75.072 |     0.982 |       0.069 |    38.025 |               35.429 |         12.089 |                 0.406 |        2271.74  |
| HistGradientBoosting | ML       | Bhavesh |     43.035 |      75.758 |     0.982 |       0.07  |    38.76  |               33.901 |          9.432 |                11.967 |        4256.6   |
| LightGBM             | ML       | Anirudh |     47.483 |      80.376 |     0.98  |       0.078 |    43.159 |               27.07  |         35.963 |                 0.977 |         214.671 |
| DecisionTree         | ML       | Ajay    |     53.292 |      93.902 |     0.972 |       0.087 |    46.944 |               18.146 |          4.134 |                 0.118 |         432.454 |
| Ridge                | baseline | Ajay    |     65.107 |     106.622 |     0.964 |       0.107 |    55.979 |                0     |          0.081 |                 0.365 |           2.56  |
| SeasonalNaive_week   | baseline | Ajay    |     75.366 |     146.112 |     0.933 |       0.123 |    56.484 |              -15.758 |          0     |                 0.001 |           0.077 |
| PCA_Ridge            | ML       | Pragun  |     76.827 |     124.501 |     0.951 |       0.126 |    68.697 |              -18.001 |          0.12  |                 0.361 |           6.606 |
| HistoricalAverage    | baseline | Ajay    |    105.338 |     179.001 |     0.899 |       0.172 |    95.792 |              -61.793 |          0.009 |                 0.046 |          41.706 |
| Persistence          | baseline | Ajay    |    109.947 |     185.263 |     0.892 |       0.18  |   102.318 |              -68.872 |          0     |                 0.001 |           0.073 |
| SeasonalNaive_day    | baseline | Ajay    |    117.227 |     234.556 |     0.827 |       0.192 |   107.033 |              -80.053 |          0     |                 0.001 |           0.078 |