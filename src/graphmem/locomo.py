"""LoCoMo loader (snap-research/locomo, data/locomo10.json).

Session timestamps such as "1:56 pm on 8 May, 2023" are parsed as UTC; they are
the only source of time in the evaluation path.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_PATH = Path("data/locomo/locomo10.json")


# Category ids as used in locomo10.json / the LoCoMo QA eval code.
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


class Turn(BaseModel):
    dia_id: str
    speaker: str
    text: str
    caption: str | None = None  # BLIP caption for shared images

    def render(self) -> str:
        s = f"{self.speaker}: {self.text}"
        if self.caption:
            s += f" [shares a photo: {self.caption}]"
        return s


class Session(BaseModel):
    index: int
    time: datetime
    turns: list[Turn]


class QA(BaseModel):
    question: str
    answer: str | None  # None for adversarial (category 5)
    category: int
    evidence: list[str]
    adversarial_answer: str | None = None

    @property
    def category_name(self) -> str:
        return CATEGORIES.get(self.category, str(self.category))


class Conversation(BaseModel):
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    qa: list[QA]

    @property
    def turns(self) -> list[Turn]:
        return [t for s in self.sessions for t in s.turns]

    @property
    def last_time(self) -> datetime:
        return self.sessions[-1].time


def parse_session_time(s: str) -> datetime:
    # "1:56 pm on 8 May, 2023"  /  "10:37 am on 27 June, 2023"
    s = re.sub(r"\s+", " ", s.strip())
    return datetime.strptime(s, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)


def _parse(raw: dict) -> Conversation:
    c = raw["conversation"]
    sessions = []
    i = 1
    while f"session_{i}" in c:
        turns = c[f"session_{i}"]
        if turns:  # a few sessions are empty lists
            sessions.append(
                Session(
                    index=i,
                    time=parse_session_time(c[f"session_{i}_date_time"]),
                    turns=[
                        Turn(
                            dia_id=t["dia_id"],
                            speaker=t["speaker"],
                            text=t.get("text", ""),
                            caption=t.get("blip_caption"),
                        )
                        for t in turns
                    ],
                )
            )
        i += 1
    qa = [
        QA(
            question=q["question"],
            answer=None if q.get("answer") is None else str(q["answer"]),
            category=q["category"],
            evidence=q.get("evidence", []),
            adversarial_answer=q.get("adversarial_answer"),
        )
        for q in raw["qa"]
    ]
    return Conversation(
        sample_id=raw["sample_id"],
        speaker_a=c["speaker_a"],
        speaker_b=c["speaker_b"],
        sessions=sessions,
        qa=qa,
    )


def load_locomo(path: str | Path = DEFAULT_PATH) -> list[Conversation]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; download it with:\n  curl -L -o {p} {LOCOMO_URL}")
    return [_parse(r) for r in json.loads(p.read_text(encoding="utf-8"))]


def split(
    convs: list[Conversation], dev_ids: list[str] | None = None
) -> tuple[list[Conversation], list[Conversation]]:
    """Dev/test split; dev ids come from Settings.dev_conversations."""
    if dev_ids is None:
        from .config import Settings

        dev_ids = Settings().dev_conversations
    dev = [c for c in convs if c.sample_id in dev_ids]
    test = [c for c in convs if c.sample_id not in dev_ids]
    return dev, test


class TestSetGuard(RuntimeError):
    __test__ = False  # not a pytest test class


def require_dev(sample_id: str, dev_ids: list[str], allow_test: bool = False) -> None:
    """Refuse to touch a test conversation unless explicitly allowed (final eval only)."""
    if sample_id not in dev_ids and not allow_test:
        raise TestSetGuard(
            f"{sample_id} is a TEST conversation (dev = {dev_ids}). Tuning/prompt work is "
            "dev-only; pass --allow-test only for the final evaluation run."
        )
