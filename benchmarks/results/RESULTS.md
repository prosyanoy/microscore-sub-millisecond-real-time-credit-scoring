# Benchmark results

300 trees / 18300 nodes, 2 cores, Python 3.14.4, LightGBM 4.7.0. Numbers are machine-specific.

## Single-row latency (microseconds, in-process, 1 row per call)

| engine | p50 | p90 | p99 | p99.9 | mean | max abs diff vs LightGBM |
|---|---|---|---|---|---|---|
| lightgbm | 91.8 | 3106.5 | 3480.1 | 5648.9 | 606.9 | 0.0e+00 |
| lightgbm-1t | 96.7 | 124.2 | 161.9 | 403.8 | 107.0 | 0.0e+00 |
| onnxruntime | 48.9 | 66.1 | 99.8 | 462.0 | 54.7 | 3.2e-07 |
| cpp | 14.3 | 16.1 | 30.9 | 47.1 | 15.0 | 0.0e+00 |

C++ p50 is 6x lower than LightGBM's default Python path.

## Batch size 1

| engine | rows/s | p50 batch ms | p99 batch ms |
|---|---|---|---|
| lightgbm | 11,994 | 0.069 | 0.131 |
| lightgbm-1t | 11,465 | 0.077 | 0.131 |
| onnxruntime | 26,227 | 0.036 | 0.065 |
| cpp | 196,649 | 0.005 | 0.012 |
| cpp-2t | 191,791 | 0.005 | 0.021 |

## Batch size 64

| engine | rows/s | p50 batch ms | p99 batch ms |
|---|---|---|---|
| lightgbm | 63,500 | 0.899 | 3.995 |
| lightgbm-1t | 36,234 | 1.748 | 2.143 |
| onnxruntime | 85,339 | 0.732 | 1.102 |
| cpp | 97,664 | 0.628 | 1.271 |
| cpp-2t | 97,374 | 0.636 | 1.031 |

## Batch size 1024

| engine | rows/s | p50 batch ms | p99 batch ms |
|---|---|---|---|
| lightgbm | 35,458 | 28.161 | 39.603 |
| lightgbm-1t | 38,382 | 26.355 | 33.551 |
| onnxruntime | 95,714 | 10.376 | 17.878 |
| cpp | 99,023 | 10.193 | 13.770 |
| cpp-2t | 82,562 | 10.960 | 34.736 |
