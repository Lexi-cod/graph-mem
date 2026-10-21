# Results: conv-41, conv-42, conv-43, conv-44, conv-47, conv-48, conv-49, conv-50; thinking ON; judge accuracy % [95% bootstrap CI over questions]
judges: qwen3_14b, qwen3-32b-judge

| System | Judge | n | single-hop | multi-hop | temporal | open-domain | **Overall** | F1 |
|---|---|---|---|---|---|---|---|---|
| GraphMem (frozen) | qwen3_14b | 1307 | 55.8 [52.1, 59.6] (n=727) | 27.6 [22.2, 33.1] (n=239) | 59.3 [53.1, 65.1] (n=258) | 33.7 [24.1, 43.4] (n=83) | **50.0 [47.4, 52.7]** | 46.5 |
| GraphMem (frozen) | qwen3-32b-judge | 1307 | 63.1 [59.4, 66.6] (n=727) | 35.1 [29.3, 41.0] (n=239) | 70.2 [64.3, 75.6] (n=258) | 39.8 [30.1, 49.4] (n=83) | **57.9 [55.2, 60.6]** | 46.5 |
| Mem0 (k10) | qwen3_14b | 1307 | 59.7 [56.1, 63.1] (n=727) | 25.9 [20.1, 31.4] (n=239) | 46.1 [40.3, 51.9] (n=258) | 25.3 [16.9, 34.9] (n=83) | **48.7 [46.0, 51.3]** | 44.1 |
| Mem0 (k10) | qwen3-32b-judge | 1307 | 69.9 [66.7, 73.0] (n=727) | 32.6 [26.8, 38.5] (n=239) | 60.1 [54.3, 65.9] (n=258) | 36.1 [26.5, 47.0] (n=83) | **59.0 [56.3, 61.6]** | 44.1 |
| Mem0 token-matched (k24) | qwen3_14b | 1307 | 64.5 [61.2, 67.8] (n=727) | 33.9 [28.0, 39.7] (n=239) | 50.0 [43.8, 56.2] (n=258) | 25.3 [16.9, 34.9] (n=83) | **53.6 [50.9, 56.2]** | 47.3 |
| Mem0 token-matched (k24) | qwen3-32b-judge | 1307 | 73.5 [70.4, 76.5] (n=727) | 43.1 [36.4, 49.4] (n=239) | 66.7 [60.9, 72.1] (n=258) | 34.9 [25.3, 44.6] (n=83) | **64.1 [61.5, 66.6]** | 47.3 |
| FAISS (k10) | qwen3_14b | 1307 | 45.3 [41.5, 48.7] (n=727) | 16.3 [11.7, 21.3] (n=239) | 39.9 [34.1, 45.7] (n=258) | 28.9 [19.3, 38.6] (n=83) | **37.9 [35.3, 40.6]** | 36.8 |
| FAISS (k10) | qwen3-32b-judge | 1307 | 53.1 [49.4, 56.5] (n=727) | 20.5 [15.5, 25.9] (n=239) | 48.1 [41.9, 53.9] (n=258) | 32.5 [22.9, 42.2] (n=83) | **44.8 [42.2, 47.5]** | 36.8 |
| FAISS token-matched (k24) | qwen3_14b | 1307 | 51.6 [48.0, 55.2] (n=727) | 25.9 [20.5, 31.4] (n=239) | 46.9 [40.7, 52.7] (n=258) | 31.3 [21.7, 41.0] (n=83) | **44.7 [42.2, 47.3]** | 42.7 |
| FAISS token-matched (k24) | qwen3-32b-judge | 1307 | 60.1 [56.5, 63.4] (n=727) | 31.4 [25.5, 37.2] (n=239) | 57.0 [51.2, 62.8] (n=258) | 36.1 [26.5, 45.8] (n=83) | **52.7 [50.2, 55.4]** | 42.7 |

## Retrieval latency and answer step (thinking on)
| System | retrieval-only p50 / p95 ms | end-to-end retrieval p50 / p95 ms | answer s/q | answer out tok | prompt tok |
|---|---|---|---|---|---|
| GraphMem (frozen) | 47.3 / 71.6 | 64.2 / 94.2 | 7.11 | 424 | 1314 |
| Mem0 (k10) | 42.3 / 72.1 | 140.4 / 226.4 | 6.40 | 428 | 676 |
| Mem0 token-matched (k24) | 41.1 / 74.0 | 134.6 / 223.7 | 7.36 | 448 | 1271 |
| FAISS (k10) | 0.2 / 0.3 | 16.4 / 28.4 | 6.78 | 456 | 700 |
| FAISS token-matched (k24) | 0.1 / 0.3 | 13.8 / 24.5 | 7.84 | 477 | 1329 |

judge agreement on final-table answers: 4720 triples, raw 90.0%, Cohen's kappa 0.799; qwen3_14b C-rate 38.9%, qwen3-32b-judge C-rate 48.6%; confusion (qwen3_14/qwen3-32): {'CC': 1829, 'WW': 2419, 'WC': 464, 'CW': 8}

judge agreement on all dev answers: 4720 triples, raw 90.0%, Cohen's kappa 0.799; qwen3_14b C-rate 38.9%, qwen3-32b-judge C-rate 48.6%; confusion (qwen3_14/qwen3-32): {'CC': 1829, 'WW': 2419, 'WC': 464, 'CW': 8}

wrote results\phase4\judge_sample_50_test.md (5 of 50 rows where the judges disagree)
