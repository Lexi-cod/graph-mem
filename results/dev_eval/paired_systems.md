# Paired bootstrap over questions (conv-26 + conv-30, thinking on): A minus B, accuracy points
10,000 resamples; [95% CI]; p = two-sided bootstrap p-value

## GraphMem (lambda=0) vs Mem0, judge qwen3_14b (n=233)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 114 | 38.6 | 58.8 | -20.2 [-30.7, -9.6] | 0.000 |
| multi-hop | 43 | 23.3 | 23.3 | +0.0 [-16.3, +16.3] | 1.000 |
| temporal | 63 | 68.3 | 61.9 | +6.3 [-7.9, +20.6] | 0.430 |
| open-domain | 13 | 46.2 | 61.5 | -15.4 [-38.5, +0.0] | 0.237 |
| overall | 233 | 44.2 | 53.2 | -9.0 [-16.3, -1.7] | 0.018 |

## GraphMem (lambda=0) vs Mem0, judge qwen3-32b-judge (n=233)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 114 | 46.5 | 67.5 | -21.1 [-31.6, -10.5] | 0.000 |
| multi-hop | 43 | 34.9 | 32.6 | +2.3 [-16.3, +18.6] | 0.916 |
| temporal | 63 | 79.4 | 73.0 | +6.3 [-6.3, +19.0] | 0.367 |
| open-domain | 13 | 61.5 | 76.9 | -15.4 [-38.5, +0.0] | 0.227 |
| overall | 233 | 54.1 | 63.1 | -9.0 [-16.3, -1.7] | 0.017 |

## GraphMem (lambda=0) vs FAISS RAG, judge qwen3_14b (n=233)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 114 | 38.6 | 48.2 | -9.6 [-21.1, +1.8] | 0.115 |
| multi-hop | 43 | 23.3 | 9.3 | +14.0 [+0.0, +30.2] | 0.099 |
| temporal | 63 | 68.3 | 55.6 | +12.7 [+0.0, +27.0] | 0.081 |
| open-domain | 13 | 46.2 | 38.5 | +7.7 [-15.4, +30.8] | 0.779 |
| overall | 233 | 44.2 | 42.5 | +1.7 [-6.0, +9.0] | 0.700 |

## GraphMem (lambda=0) vs FAISS RAG, judge qwen3-32b-judge (n=233)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 114 | 46.5 | 53.5 | -7.0 [-18.4, +5.3] | 0.283 |
| multi-hop | 43 | 34.9 | 16.3 | +18.6 [+4.7, +32.6] | 0.017 |
| temporal | 63 | 79.4 | 61.9 | +17.5 [+4.8, +31.7] | 0.015 |
| open-domain | 13 | 61.5 | 61.5 | +0.0 [-30.8, +30.8] | 1.000 |
| overall | 233 | 54.1 | 49.4 | +4.7 [-3.0, +12.4] | 0.256 |

