# Slice 9 — Offline Ragas eval + CI gate

Offline, server-independent. Calls the pipeline functions directly (not the HTTP
API), against the _test DB seeded with the golden source docs.

Eval script (eval/run_eval.py):
- Load eval/golden/golden.jsonl.
- Ingest the golden source docs into the _test DB (reuse the ingestion code).
- For each question: run the real retrieval (hybrid + rerank) and generation,
  collect (question, answer, retrieved_contexts, ground_truth).
- Score with Ragas: faithfulness, context_precision, answer_relevancy.
  Judge LLM + embeddings = Gemini (free tier). Confirm current Ragas API.
- Print per-metric averages. Optionally write a row to eval_runs / eval_results
  (see docs/04) with git sha + summary.
- Gate: thresholds from config (e.g. faithfulness>=0.85, context_precision>=0.70).
  Exit non-zero if any metric is below its threshold.

Rerank-impact flag: run with the slice-5 "disable rerank" flag on and off, so you
can report what reranking actually buys. (This comparison is a portfolio artifact.)

CI (.github/workflows/eval.yml):
- On pull_request: spin up a ParadeDB service container + Redis, run migrations,
  run eval/run_eval.py, fail the job if it exits non-zero.
- GEMINI_API_KEY from GitHub Actions secrets. Cache the uv env + the torch/
  sentence-transformers download (reranker) to keep CI fast.
- Cost note: LLM-judge calls run per PR. Keep the PR set small; optionally run
  the FULL golden set on a nightly schedule and a small smoke subset on PRs.

Tests:
- run_eval on a tiny 2-3 item fixture set produces scores and a pass/fail.
- the gate exits non-zero when a metric is below threshold (feed a deliberately
  bad answer).
- eval never touches the dev DB (uses the _test guard from conftest).

What can go wrong: Gemini 429 during judging (throttle/retry, small batches),
non-deterministic scores (fix temperature=0, average over the set), golden docs
not ingested before scoring.