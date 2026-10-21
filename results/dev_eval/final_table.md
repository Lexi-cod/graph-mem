# Results: conv-26, conv-30; thinking ON; judge accuracy % [95% bootstrap CI over questions]
judges: qwen3_14b, qwen3-32b-judge

| System | Judge | n | single-hop | multi-hop | temporal | open-domain | **Overall** | F1 |
|---|---|---|---|---|---|---|---|---|
| FAISS RAG | qwen3_14b | 233 | 48.2 [39.5, 57.9] (n=114) | 9.3 [2.3, 18.6] (n=43) | 55.6 [42.9, 68.3] (n=63) | 38.5 [15.4, 69.2] (n=13) | **42.5 [36.5, 48.5]** | 37.4 |
| FAISS RAG | qwen3-32b-judge | 233 | 53.5 [43.9, 62.3] (n=114) | 16.3 [7.0, 27.9] (n=43) | 61.9 [50.8, 73.0] (n=63) | 61.5 [38.5, 84.6] (n=13) | **49.4 [42.9, 55.4]** | 37.4 |
| GraphMem (lambda=0) | qwen3_14b | 233 | 38.6 [29.8, 48.2] (n=114) | 23.3 [11.6, 37.2] (n=43) | 68.3 [57.1, 79.4] (n=63) | 46.2 [23.1, 69.2] (n=13) | **44.2 [37.8, 50.6]** | 40.6 |
| GraphMem (lambda=0) | qwen3-32b-judge | 233 | 46.5 [37.7, 56.1] (n=114) | 34.9 [20.9, 48.8] (n=43) | 79.4 [69.8, 88.9] (n=63) | 61.5 [38.5, 84.6] (n=13) | **54.1 [47.6, 60.5]** | 40.6 |
| GraphMem + provenance | qwen3_14b | 233 | 51.8 [43.0, 61.4] (n=114) | 27.9 [14.0, 41.9] (n=43) | 71.4 [60.3, 82.5] (n=63) | 61.5 [38.5, 84.6] (n=13) | **53.2 [46.8, 59.7]** | 49.2 |
| GraphMem + provenance | qwen3-32b-judge | 233 | 60.5 [51.8, 69.3] (n=114) | 34.9 [20.9, 48.8] (n=43) | 85.7 [76.2, 93.7] (n=63) | 76.9 [53.8, 100.0] (n=13) | **63.5 [57.5, 69.5]** | 49.2 |
| Mem0 | qwen3_14b | 233 | 58.8 [50.0, 67.5] (n=114) | 23.3 [11.6, 37.2] (n=43) | 61.9 [50.8, 73.0] (n=63) | 61.5 [38.5, 84.6] (n=13) | **53.2 [46.8, 59.2]** | 41.1 |
| Mem0 | qwen3-32b-judge | 233 | 67.5 [58.8, 76.3] (n=114) | 32.6 [18.6, 46.5] (n=43) | 73.0 [61.9, 84.1] (n=63) | 76.9 [53.8, 100.0] (n=13) | **63.1 [57.1, 69.1]** | 41.1 |

## Retrieval latency and answer step (thinking on)
| System | retrieval-only p50 / p95 ms | end-to-end retrieval p50 / p95 ms | answer s/q | answer out tok | prompt tok |
|---|---|---|---|---|---|
| FAISS RAG | 0.2 / 0.3 | 31.1 / 39.7 | 6.22 | 414 | 683 |
| GraphMem (lambda=0) | 69.7 / 132.6 | 102.9 / 167.1 | 5.48 | 361 | 622 |
| GraphMem + provenance | 72.9 / 102.2 | 105.5 / 142.4 | 6.31 | 366 | 1287 |
| Mem0 | 62.2 / 73.9 | 205.5 / 285.6 | 6.89 | 455 | 691 |

judge agreement on final-table answers: 747 triples, raw 89.6%, Cohen's kappa 0.792; qwen3_14b C-rate 41.1%, qwen3-32b-judge C-rate 51.0%; confusion (qwen3_14/qwen3-32): {'WW': 364, 'CC': 305, 'WC': 76, 'CW': 2}

judge agreement on all dev answers: 1581 triples, raw 90.4%, Cohen's kappa 0.789; qwen3_14b C-rate 29.6%, qwen3-32b-judge C-rate 38.5%; confusion (qwen3_14/qwen3-32): {'WW': 967, 'CC': 463, 'WC': 146, 'CW': 5}

results\dev_eval\nul exists; not overwritten (it may hold your hand-check marks). Use --sample-out NAME or --overwrite-sample.
