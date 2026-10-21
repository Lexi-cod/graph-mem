"""Shared QA machinery: answer prompt, LLM judge, token F1. Identical for every system,
so the only thing that differs between systems is the retrieved context."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from ..llm import LLM

ANSWER_SYSTEM = """You answer questions about a long conversation between two people,
using only the retrieved memories provided. Memories are facts or conversation
excerpts, each marked with the date it was said; "no longer true" marks outdated facts.
Answer with a short phrase (a few words), not a full sentence. If the memories do not
contain the answer, give your best short guess.

For "when" questions, resolve relative times against the date shown on the memory
that mentions them, and answer with the resolved date or period. Examples: a memory
dated 15 July 2023 saying "last Friday" -> "the Friday before 15 July 2023"; dated
8 May 2023 saying "yesterday" -> "7 May 2023"; dated 8 May 2023 saying "last year" -> "2022".
Never answer with the memory's own date when it refers to a different time."""


def answer_prompt(question: str, context: str) -> str:
    return f"Memories:\n{context or '(none)'}\n\nQuestion: {question}\nShort answer:"


class AnswerResult(BaseModel):
    text: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float  # original request latency (also for cache hits)
    cached: bool


async def answer(llm: LLM, question: str, context: str, *, thinking: bool = False,
                 max_tokens: int = 64, thinking_max_tokens: int = 4096) -> AnswerResult:
    """The shared answer step. `thinking` turns Qwen3 reasoning on for this call only."""
    r = await llm.generate(answer_prompt(question, context), system=ANSWER_SYSTEM,
                           purpose="answer_thinking" if thinking else "answer",
                           max_output_tokens=thinking_max_tokens if thinking else max_tokens,
                           thinking=thinking)
    text = _THINK.sub("", r.text).strip()
    return AnswerResult(text=text, prompt_tokens=r.input_tokens, output_tokens=r.output_tokens,
                        latency_s=r.latency_s, cached=r.cached)


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


JUDGE_SYSTEM = """You grade answers to questions about a conversation.
Given the question, the gold answer and a generated answer, output CORRECT if the
generated answer conveys the same information as the gold answer, else WRONG.
Be generous with wording and format: paraphrases, partial dates in another format
("May 7, 2023" vs "7 May 2023"), or relative phrases resolving to the same date or
period count as CORRECT. A generated answer that contains the gold answer plus extra
non-contradictory detail is CORRECT. Missing or different key information is WRONG."""


class Verdict(BaseModel):
    label: Literal["CORRECT", "WRONG"]


async def judge(llm: LLM, question: str, gold: str, generated: str, purpose: str = "judge") -> bool:
    v = await llm.generate_json(
        f"Question: {question}\nGold answer: {gold}\nGenerated answer: {generated}",
        Verdict, system=JUDGE_SYSTEM, purpose=purpose, max_output_tokens=32)
    return v.label == "CORRECT"


def _normalize(s: str) -> list[str]:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return s.split()


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token F1 (lowercase, no punctuation/articles; no stemming)."""
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)
