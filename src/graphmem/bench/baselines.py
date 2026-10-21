"""Baseline retrieval sizes for Phase 4 (part of the frozen config).

k=10 matches GraphMem's fact k. The "matched" sizes give FAISS and Mem0 the same mean
answer-prompt size as GraphMem+provenance on dev (1,287 tokens; Qwen3 tokenizer,
calibrated to the server's count), chosen by scripts/token_match.py from retrieval only.
"""

BASELINES = {
    "faiss_k": 10,
    "mem0_k": 10,
    "faiss_matched_k": 24,  # 1,312 tokens on dev (+2.0%); results/dev_eval/token_match.json
    "mem0_matched_k": 24,  # 1,309 tokens on dev (+1.7%)
}
