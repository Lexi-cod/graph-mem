# Paired bootstrap over questions (conv-41, conv-42, conv-43, conv-44, conv-47, conv-48, conv-49, conv-50; thinking on): A minus B, accuracy points
10,000 resamples; [95% CI]; p = two-sided bootstrap p-value

## GraphMem + provenance vs mem0:k=10, judge qwen3_14b (n=1307)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 727 | 55.8 | 59.7 | -3.9 [-7.7, +0.0] | 0.056 |
| multi-hop | 239 | 27.6 | 25.9 | +1.7 [-3.8, +7.1] | 0.588 |
| temporal | 258 | 59.3 | 46.1 | +13.2 [+7.0, +19.8] | 0.000 |
| open-domain | 83 | 33.7 | 25.3 | +8.4 [+2.4, +15.7] | 0.016 |
| overall | 1307 | 50.0 | 48.7 | +1.3 [-1.4, +4.1] | 0.357 |

## GraphMem + provenance vs mem0:k=10, judge qwen3-32b-judge (n=1307)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 727 | 63.1 | 69.9 | -6.7 [-10.5, -3.0] | 0.001 |
| multi-hop | 239 | 35.1 | 32.6 | +2.5 [-3.8, +8.8] | 0.456 |
| temporal | 258 | 70.2 | 60.1 | +10.1 [+3.9, +16.3] | 0.002 |
| open-domain | 83 | 39.8 | 36.1 | +3.6 [-6.0, +13.3] | 0.520 |
| overall | 1307 | 57.9 | 59.0 | -1.1 [-3.8, +1.8] | 0.453 |

## GraphMem + provenance vs mem0:k=24, judge qwen3_14b (n=1307)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 727 | 55.8 | 64.5 | -8.7 [-12.7, -4.7] | 0.000 |
| multi-hop | 239 | 27.6 | 33.9 | -6.3 [-12.1, -0.4] | 0.045 |
| temporal | 258 | 59.3 | 50.0 | +9.3 [+2.7, +16.3] | 0.007 |
| open-domain | 83 | 33.7 | 25.3 | +8.4 [+1.2, +16.9] | 0.037 |
| overall | 1307 | 50.0 | 53.6 | -3.6 [-6.4, -0.8] | 0.016 |

## GraphMem + provenance vs mem0:k=24, judge qwen3-32b-judge (n=1307)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 727 | 63.1 | 73.5 | -10.3 [-14.2, -6.5] | 0.000 |
| multi-hop | 239 | 35.1 | 43.1 | -7.9 [-14.6, -1.7] | 0.018 |
| temporal | 258 | 70.2 | 66.7 | +3.5 [-2.7, +9.7] | 0.277 |
| open-domain | 83 | 39.8 | 34.9 | +4.8 [-4.8, +14.5] | 0.365 |
| overall | 1307 | 57.9 | 64.1 | -6.2 [-9.0, -3.4] | 0.000 |

## GraphMem + provenance vs faiss:k=10, judge qwen3_14b (n=1307)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 727 | 55.8 | 45.3 | +10.6 [+6.5, +14.9] | 0.000 |
| multi-hop | 239 | 27.6 | 16.3 | +11.3 [+5.9, +16.7] | 0.000 |
| temporal | 258 | 59.3 | 39.9 | +19.4 [+12.4, +26.4] | 0.000 |
| open-domain | 83 | 33.7 | 28.9 | +4.8 [-2.4, +13.3] | 0.303 |
| overall | 1307 | 50.0 | 37.9 | +12.1 [+9.2, +15.1] | 0.000 |

## GraphMem + provenance vs faiss:k=10, judge qwen3-32b-judge (n=1307)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 727 | 63.1 | 53.1 | +10.0 [+5.9, +14.3] | 0.000 |
| multi-hop | 239 | 35.1 | 20.5 | +14.6 [+8.8, +20.5] | 0.000 |
| temporal | 258 | 70.2 | 48.1 | +22.1 [+15.5, +28.7] | 0.000 |
| open-domain | 83 | 39.8 | 32.5 | +7.2 [+0.0, +15.7] | 0.095 |
| overall | 1307 | 57.9 | 44.8 | +13.1 [+10.1, +16.1] | 0.000 |

## GraphMem + provenance vs faiss:k=24, judge qwen3_14b (n=1307)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 727 | 55.8 | 51.6 | +4.3 [+0.0, +8.5] | 0.056 |
| multi-hop | 239 | 27.6 | 25.9 | +1.7 [-3.8, +7.1] | 0.602 |
| temporal | 258 | 59.3 | 46.9 | +12.4 [+5.8, +19.0] | 0.000 |
| open-domain | 83 | 33.7 | 31.3 | +2.4 [-6.0, +10.8] | 0.673 |
| overall | 1307 | 50.0 | 44.7 | +5.3 [+2.4, +8.3] | 0.001 |

## GraphMem + provenance vs faiss:k=24, judge qwen3-32b-judge (n=1307)
| category | n | A acc | B acc | A-B [95% CI] | p |
|---|---|---|---|---|---|
| single-hop | 727 | 63.1 | 60.1 | +3.0 [-1.1, +7.2] | 0.159 |
| multi-hop | 239 | 35.1 | 31.4 | +3.8 [-2.1, +9.6] | 0.236 |
| temporal | 258 | 70.2 | 57.0 | +13.2 [+6.6, +20.2] | 0.000 |
| open-domain | 83 | 39.8 | 36.1 | +3.6 [-3.6, +12.0] | 0.444 |
| overall | 1307 | 57.9 | 52.7 | +5.2 [+2.3, +8.2] | 0.000 |

