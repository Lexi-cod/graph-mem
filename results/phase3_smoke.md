# Phase 3 smoke test (real stack)

- API: `uvicorn graphmem.api:app` on port 8765; conversation `smoke-test` (cleared first)
- Neo4j: bolt://localhost:7687; LLM: `qwen3:14b` at http://localhost:11435/v1 (answer thinking on, extraction/reason thinking off)
- Frozen retrieval config: k=10, provenance=True, contradiction_mode=rules, prune_enabled=False
- Message timestamps are explicit (start 2026-10-21T09:00:00+00:00), one minute apart.

`GET /health` -> {'status': 'ok'}

## Agent turns (`POST /agent/turn`)

### Turn 1: Caroline
> Hi Mel! Big news: I adopted a guinea pig named Oscar last weekend. He loves parsley.

- **remembered (mid-run write):** True — The message contains lasting information about Caroline adopting a guinea pig named Oscar and his preference for parsley, which are facts that should be remembered.
- **written facts:** ['Caroline adopted a guinea pig named Oscar on the weekend before 21 October 2026.', 'Oscar loves parsley.']
- write stats: created 2, reinforced 0, retired 0, entities created 3, merged 0; dropped []
- retrieved facts used: 2; prune: None
- timings (ms): {'retrieve': 17080.1, 'reason': 636.1, 'write': 3600.6, 'answer': 2222.5}; round trip 23577 ms
- **answer:** Hi Caroline! That's wonderful news—Oscar sounds like a delightful addition to your life. It's great that he loves parsley; I'm sure he'll enjoy plenty of fresh greens! 🐹 parsley 🌿

### Turn 2: Melanie
> Aww, congrats! How is Oscar settling in?

- **remembered (mid-run write):** False — 
- retrieved facts used: 2; prune: None
- timings (ms): {'retrieve': 48.7, 'reason': 194.5, 'answer': 3406.1}; round trip 3665 ms
- **answer:** Caroline: Oscar is doing great! He’s already taken a liking to parsley, which is his favorite. He’s been exploring his new home and seems happy. 🐹✨

### Turn 3: Caroline
> He's great. I also signed up for a pottery class at the community center that meets every Tuesday.

- **remembered (mid-run write):** True — The message contains new information about Caroline's actions, specifically that she signed up for a pottery class at the community center that meets every Tuesday. This is a lasting fact about her plans and activities that should be remembered.
- **written facts:** ['Caroline signed up for a pottery class at the community center that meets every Tuesday.']
- write stats: created 1, reinforced 0, retired 0, entities created 1, merged 1; dropped [{'reason': 'evidence_not_in_turn', 'fact': 'Caroline adopted a guinea pig named Oscar on Sunday 17 October 2026.'}, {'reason': 'subject_not_in_turn', 'fact': 'The pottery class that meets every Tuesday is located at the community center.'}]
- retrieved facts used: 3; prune: None
- timings (ms): {'retrieve': 63.1, 'reason': 714.3, 'write': 4215.8, 'answer': 4185.9}; round trip 9234 ms
- **answer:** That's exciting! I hope your pottery class is a lot of fun. Oh, and I heard Oscar (your guinea pig) loves parsley—maybe you can bring some snacks for him next time you're at the community center! 🐹

### Turn 4: Caroline
> One more thing: I moved from Boston to Seattle last month to start a new job as a counselor at a youth center.

- **remembered (mid-run write):** True — The message provides lasting information about Caroline's move from Boston to Seattle and her new job as a counselor at a youth center, which are significant life changes and should be remembered.
- **written facts:** ['Caroline moved from Boston to Seattle in September 2026.', 'Caroline started a new job as a counselor at a youth center in September 2026.']
- write stats: created 2, reinforced 0, retired 0, entities created 2, merged 1; dropped []
- retrieved facts used: 5; prune: None
- timings (ms): {'retrieve': 52.9, 'reason': 634.5, 'write': 3232.7, 'answer': 4537.1}; round trip 8476 ms
- **answer:** That's exciting, Caroline! I'm glad you're settled into your new job as a counselor in Seattle. Have you started the pottery class yet?

### Turn 5: Melanie
> Wow, lots of changes! So what's new with you overall?

- **remembered (mid-run write):** False — 
- retrieved facts used: 5; prune: None
- timings (ms): {'retrieve': 110.8, 'reason': 217.6, 'answer': 2194.3}; round trip 2543 ms
- **answer:** Hi Mel! Since moving to Seattle in September, I started a new job as a counselor at a youth center. I also adopted a guinea pig named Oscar, who adores parsley, and signed up for a pottery class that meets every Tuesday. Lots of new adventures! 🐹坯

## Memory queries (`POST /memory/query`)

### What pet does Caroline have and what does it like?

- db 32.51 ms, embed 34.24 ms; 5 facts, 3 source turns
```text
Facts:
- Caroline adopted a guinea pig named Oscar on the weekend before 21 October 2026. [stated 2026-10-21]
- Caroline moved from Boston to Seattle in September 2026. [stated 2026-10-21]
- Caroline signed up for a pottery class at the community center that meets every Tuesday. [stated 2026-10-21]
- Caroline started a new job as a counselor at a youth center in September 2026. [stated 2026-10-21]
- Oscar loves parsley. [stated 2026-10-21]

Source turns (where these facts were said):
- [21 October 2026] Caroline: Hi Mel! Big news: I adopted a guinea pig named Oscar last weekend. He loves parsley.
- [21 October 2026] Caroline: One more thing: I moved from Boston to Seattle last month to start a new job as a counselor at a youth center.
- [21 October 2026] Caroline: He's great. I also signed up for a pottery class at the community center that meets every Tuesday.
```

### Where does Caroline live now?

- db 39.99 ms, embed 34.88 ms; 5 facts, 3 source turns
```text
Facts:
- Caroline moved from Boston to Seattle in September 2026. [stated 2026-10-21]
- Caroline started a new job as a counselor at a youth center in September 2026. [stated 2026-10-21]
- Caroline signed up for a pottery class at the community center that meets every Tuesday. [stated 2026-10-21]
- Caroline adopted a guinea pig named Oscar on the weekend before 21 October 2026. [stated 2026-10-21]
- Oscar loves parsley. [stated 2026-10-21]

Source turns (where these facts were said):
- [21 October 2026] Caroline: One more thing: I moved from Boston to Seattle last month to start a new job as a counselor at a youth center.
- [21 October 2026] Caroline: He's great. I also signed up for a pottery class at the community center that meets every Tuesday.
- [21 October 2026] Caroline: Hi Mel! Big news: I adopted a guinea pig named Oscar last weekend. He loves parsley.
```

### What class did Caroline sign up for and when does it meet?

- db 44.51 ms, embed 41.5 ms; 5 facts, 3 source turns
```text
Facts:
- Caroline signed up for a pottery class at the community center that meets every Tuesday. [stated 2026-10-21]
- Caroline started a new job as a counselor at a youth center in September 2026. [stated 2026-10-21]
- Caroline moved from Boston to Seattle in September 2026. [stated 2026-10-21]
- Caroline adopted a guinea pig named Oscar on the weekend before 21 October 2026. [stated 2026-10-21]
- Oscar loves parsley. [stated 2026-10-21]

Source turns (where these facts were said):
- [21 October 2026] Caroline: He's great. I also signed up for a pottery class at the community center that meets every Tuesday.
- [21 October 2026] Caroline: One more thing: I moved from Boston to Seattle last month to start a new job as a counselor at a youth center.
- [21 October 2026] Caroline: Hi Mel! Big news: I adopted a guinea pig named Oscar last weekend. He loves parsley.
```

## Summary

- 5 agent turns (3 with a mid-run write), 3 queries; total wall time 53 s (includes server start-up and model warm-up).
