"""Frozen configuration for test-set runs.

`write` records every effective setting, the prompt texts' hashes, and a hash of every
source file in the package. `check` compares the current state against that record and
lists any drift; the Phase 4 runner refuses to run on test conversations if anything
differs. Freezing happened after dev-only tuning (conv-26, conv-30).

    python -m graphmem.freeze write   # once, when locking the config
    python -m graphmem.freeze check
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .config import Settings

FROZEN_PATH = Path("frozen_config.json")
_PKG = Path(__file__).resolve().parent
_SECRETS = {"llm_api_key", "neo4j_password"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot(settings: Settings | None = None) -> dict[str, Any]:
    from .bench.baselines import BASELINES
    from .bench.qa import ANSWER_SYSTEM, JUDGE_SYSTEM
    from .extraction import EXTRACTOR_VERSION, SYSTEM_PROMPT
    from .ingest import CONTRADICTION_SYSTEM

    s = settings or Settings()
    sources = {str(p.relative_to(_PKG)).replace("\\", "/"): _sha(p.read_text(encoding="utf-8"))
               for p in sorted(_PKG.rglob("*.py")) if p.name != "freeze.py"}
    scripts = _PKG.parent.parent / "scripts"
    for name in ("phase4.py", "dev_eval.py"):
        if (scripts / name).exists():
            sources[f"scripts/{name}"] = _sha((scripts / name).read_text(encoding="utf-8"))
    return {
        "settings": s.model_dump(mode="json", exclude=_SECRETS),
        "baselines": dict(BASELINES),
        "extractor_version": EXTRACTOR_VERSION,
        "prompts": {"extraction": _sha(SYSTEM_PROMPT), "contradiction": _sha(CONTRADICTION_SYSTEM),
                    "answer": _sha(ANSWER_SYSTEM), "judge": _sha(JUDGE_SYSTEM)},
        "sources": sources,
    }


def diff(frozen: dict[str, Any], current: dict[str, Any]) -> list[str]:
    out = []
    for section in ("settings", "baselines", "prompts", "sources"):
        a, b = frozen.get(section, {}), current.get(section, {})
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                out.append(f"{section}.{k}: frozen={a.get(k)!r} current={b.get(k)!r}")
    if frozen.get("extractor_version") != current.get("extractor_version"):
        out.append(f"extractor_version: {frozen.get('extractor_version')} -> {current.get('extractor_version')}")
    return out


def check(path: Path = FROZEN_PATH, settings: Settings | None = None) -> list[str]:
    if not path.exists():
        return [f"{path} missing: config was never frozen"]
    return diff(json.loads(path.read_text(encoding="utf-8")), snapshot(settings))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "write":
        if FROZEN_PATH.exists() and "--force" not in sys.argv:
            sys.exit(f"{FROZEN_PATH} exists; the config is already frozen (use --force to re-freeze)")
        snap = snapshot()
        FROZEN_PATH.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {FROZEN_PATH}: {len(snap['settings'])} settings, {len(snap['sources'])} source files")
    else:
        problems = check()
        for p in problems:
            print("DRIFT", p)
        print("frozen config: OK" if not problems else f"{len(problems)} difference(s)")
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
