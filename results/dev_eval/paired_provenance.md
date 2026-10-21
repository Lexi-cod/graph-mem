# Paired bootstrap over questions (conv-26 + conv-30, thinking on): A minus B, accuracy points
10,000 resamples; [95% CI]; p = two-sided bootstrap p-value

## GraphMem + provenance vs GraphMem (lambda=0), judge qwen3_14b (n=233)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 114 | 51.8 | 38.6 | +13.2 [+5.3, +21.1] | 0.001 |
| multi-hop | 43 | 27.9 | 23.3 | +4.7 [-4.7, +14.0] | 0.447 |
| temporal | 63 | 71.4 | 68.3 | +3.2 [-3.2, +9.5] | 0.454 |
| open-domain | 13 | 61.5 | 46.2 | +15.4 [-23.1, +53.8] | 0.542 |
| overall | 233 | 53.2 | 44.2 | +9.0 [+4.3, +14.2] | 0.000 |

## GraphMem + provenance vs GraphMem (lambda=0), judge qwen3-32b-judge (n=233)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 114 | 60.5 | 46.5 | +14.0 [+7.0, +21.9] | 0.000 |
| multi-hop | 43 | 34.9 | 34.9 | +0.0 [-9.3, +9.3] | 1.000 |
| temporal | 63 | 85.7 | 79.4 | +6.3 [+0.0, +14.3] | 0.136 |
| open-domain | 13 | 76.9 | 61.5 | +15.4 [-23.1, +46.2] | 0.542 |
| overall | 233 | 63.5 | 54.1 | +9.4 [+4.7, +14.6] | 0.000 |

## GraphMem + provenance vs Mem0, judge qwen3_14b (n=233)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 114 | 51.8 | 58.8 | -7.0 [-16.7, +1.8] | 0.162 |
| multi-hop | 43 | 27.9 | 23.3 | +4.7 [-11.6, +20.9] | 0.698 |
| temporal | 63 | 71.4 | 61.9 | +9.5 [-3.2, +22.2] | 0.156 |
| open-domain | 13 | 61.5 | 61.5 | +0.0 [-30.8, +30.8] | 1.000 |
| overall | 233 | 53.2 | 53.2 | +0.0 [-6.9, +6.9] | 1.000 |

## GraphMem + provenance vs Mem0, judge qwen3-32b-judge (n=233)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 114 | 60.5 | 67.5 | -7.0 [-16.7, +2.6] | 0.175 |
| multi-hop | 43 | 34.9 | 32.6 | +2.3 [-16.3, +20.9] | 0.915 |
| temporal | 63 | 85.7 | 73.0 | +12.7 [+3.2, +23.8] | 0.017 |
| open-domain | 13 | 76.9 | 76.9 | +0.0 [-30.8, +30.8] | 1.000 |
| overall | 233 | 63.5 | 63.1 | +0.4 [-6.4, +6.9] | 0.942 |

## GraphMem + provenance vs FAISS RAG, judge qwen3_14b (n=233)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 114 | 51.8 | 48.2 | +3.5 [-7.0, +14.0] | 0.558 |
| multi-hop | 43 | 27.9 | 9.3 | +18.6 [+2.3, +34.9] | 0.033 |
| temporal | 63 | 71.4 | 55.6 | +15.9 [+3.2, +28.6] | 0.016 |
| open-domain | 13 | 61.5 | 38.5 | +23.1 [-15.4, +61.5] | 0.330 |
| overall | 233 | 53.2 | 42.5 | +10.7 [+3.4, +18.0] | 0.004 |

## GraphMem + provenance vs FAISS RAG, judge qwen3-32b-judge (n=233)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 114 | 60.5 | 53.5 | +7.0 [-3.5, +17.5] | 0.232 |
| multi-hop | 43 | 34.9 | 16.3 | +18.6 [+4.7, +32.6] | 0.019 |
| temporal | 63 | 85.7 | 61.9 | +23.8 [+12.7, +34.9] | 0.000 |
| open-domain | 13 | 76.9 | 61.5 | +15.4 [-23.1, +53.8] | 0.539 |
| overall | 233 | 63.5 | 49.4 | +14.2 [+6.9, +21.5] | 0.000 |

