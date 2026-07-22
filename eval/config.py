"""Eval-only knobs. Kept out of `minddrill.config.Settings` — nothing in the
served app reads these; only `eval/run_eval.py` does.
"""

import os

GOLDEN_PATH = "eval/golden/golden.jsonl"

# Gate thresholds (docs/02, docs/DECISIONS.md): fail the build below these.
FAITHFULNESS_MIN = float(os.environ.get("EVAL_FAITHFULNESS_MIN", "0.85"))
CONTEXT_PRECISION_MIN = float(os.environ.get("EVAL_CONTEXT_PRECISION_MIN", "0.70"))

# A fixed, small subset for the PR gate: cheap enough to run on every push,
# still touching every question category including a decline case. The full
# golden set (`--full`) is a local/manual run, not part of the PR gate.
SMOKE_QUESTIONS = frozenset(
    {
        "How many PTO days do full-time employees get per year?",
        "How much sick leave is provided, and does it carry over?",
        "What is Northwind's 401(k) match?",
        "Is multi-factor authentication required?",
        "What are the four data classification levels?",
        "Where is Northwind Robotics headquartered?",
        "What is Northwind's bereavement leave policy?",
        "Does Northwind offer tuition reimbursement?",
    }
)

JUDGE_CHAT_MODEL = "gemini-2.5-flash"
JUDGE_EMBED_MODEL = "models/gemini-embedding-001"

# Free-tier gemini-2.5-flash is commonly capped around 10 RPM (per-project,
# viewable in the AI Studio console — Google no longer publishes a fixed
# table). 6.5s spaces requests to ~9/min, leaving headroom. This only guards
# against per-minute 429s; a per-day cap (we've seen as low as 20 req/day on
# a fresh key) isn't fixable by spacing — only by waiting for the daily reset
# or judging with a different provider.
REQUEST_DELAY_SECONDS = float(os.environ.get("EVAL_REQUEST_DELAY_SECONDS", "6.5"))
