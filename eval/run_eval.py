"""Offline Ragas eval + CI gate (slice 9).

Server-independent: calls the pipeline functions directly (ingest, hybrid
retrieve + rerank, generate), never the HTTP API, against the `_test` database
derived from `DATABASE_URL` — the same guard `tests/conftest.py` uses, so this
script can never touch the dev database.

Usage:
    uv run python eval/run_eval.py            # full golden set
    uv run python eval/run_eval.py --smoke     # small PR-gate subset
    uv run python eval/run_eval.py --out DIR   # report output directory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import structlog
from sqlalchemy import select

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eval import _ragas_compat  # noqa: E402

_ragas_compat.patch()  # must run before the first `import ragas`

from eval.config import (  # noqa: E402
    CONTEXT_PRECISION_MIN,
    FAITHFULNESS_MIN,
    GOLDEN_PATH,
    JUDGE_CHAT_MODEL,
    JUDGE_EMBED_MODEL,
    REQUEST_DELAY_SECONDS,
    SMOKE_QUESTIONS,
)

log = structlog.get_logger(__name__)

_PLACEHOLDER_CONTEXT = "(no relevant context retrieved)"


@dataclass
class GoldenItem:
    question: str
    ground_truth: str
    source_doc: str
    category: str
    answerable: bool


@dataclass
class EvalSample:
    question: str
    answer: str
    retrieved_contexts: list[str]
    ground_truth: str
    category: str
    answerable: bool
    declined: bool


def load_golden(path: str | Path) -> list[GoldenItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                GoldenItem(
                    question=row["question"],
                    ground_truth=row["ground_truth"],
                    source_doc=row["source_doc"],
                    category=row["category"],
                    answerable=row["answerable"],
                )
            )
    return items


def select_subset(items: list[GoldenItem], smoke: bool) -> list[GoldenItem]:
    if not smoke:
        return items
    return [item for item in items if item.question in SMOKE_QUESTIONS]


async def ensure_eval_user(session, username: str = "eval-runner") -> uuid.UUID:
    """Create (or reuse) the single synthetic user the eval ingests/queries as.

    `documents.user_id` and `sessions.user_id` are real foreign keys to
    `users`, unlike the denormalized plain-uuid `chunks.user_id` — so a real
    row is required, not just any UUID.
    """
    from minddrill.models.user import User

    existing = await session.scalar(select(User).where(User.username == username))
    if existing is not None:
        return existing.id
    user = User(username=username, password_hash="unused-eval-only-account")
    session.add(user)
    await session.flush()
    return user.id


async def ingest_golden(
    session, user_id: uuid.UUID, items: list[GoldenItem], embedder
) -> None:
    """Ingest every unique source doc referenced by `items` into the _test DB."""
    from minddrill.rag.ingest import ingest_pdf
    from minddrill.rag.schemas import IngestRequest

    seen: set[str] = set()
    for item in items:
        if item.source_doc in seen:
            continue
        seen.add(item.source_doc)
        source_type = "markdown" if item.source_doc.endswith(".md") else "pdf"
        req = IngestRequest(source_type=source_type, source_uri=item.source_doc)
        result = await ingest_pdf(session, user_id, req, embedder)
        log.info(
            "eval.ingested",
            source_doc=item.source_doc,
            document_id=str(result.document_id),
            chunk_count=result.chunk_count,
        )


async def build_samples(
    session,
    user_id: uuid.UUID,
    items: list[GoldenItem],
    embedder,
    providers,
    reranker,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> list[EvalSample]:
    """Drive the real pipeline (retrieve + rerank + generate) per question."""
    from minddrill.models.chunk import Chunk
    from minddrill.providers.failover import open_stream
    from minddrill.rag.retrieve import prepare_answer

    samples: list[EvalSample] = []
    for i, item in enumerate(items):
        request_id = f"eval-{i}"
        plan = await prepare_answer(
            session, user_id, item.question, embedder, reranker, request_id
        )
        if plan.decline_reason is not None:
            samples.append(
                EvalSample(
                    question=item.question,
                    answer=plan.decline_reason,
                    retrieved_contexts=[_PLACEHOLDER_CONTEXT],
                    ground_truth=item.ground_truth,
                    category=item.category,
                    answerable=item.answerable,
                    declined=True,
                )
            )
            continue

        # `prepare_answer` guarantees sources/messages are set whenever
        # decline_reason is None; assert narrows this for the type checker.
        assert plan.sources is not None
        assert plan.messages is not None

        chunk_ids = [uuid.UUID(s["chunk_id"]) for s in plan.sources]
        rows = await session.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids)))
        by_id = {c.id: c.content for c in rows}
        contexts = [by_id[cid] for cid in chunk_ids if cid in by_id] or [
            _PLACEHOLDER_CONTEXT
        ]

        tokens, _provider = await open_stream(providers, plan.messages)
        answer_parts = [tok async for tok in tokens]
        aclose = getattr(tokens, "aclose", None)
        if aclose is not None:
            await aclose()

        # Space out generate calls to stay under the free-tier per-minute cap.
        # Skipped after the last item, and callers (e.g. tests, against a fake
        # in-memory provider) can pass request_delay=0 to skip it entirely.
        if request_delay and i < len(items) - 1:
            await asyncio.sleep(request_delay)

        samples.append(
            EvalSample(
                question=item.question,
                answer="".join(answer_parts),
                retrieved_contexts=contexts,
                ground_truth=item.ground_truth,
                category=item.category,
                answerable=item.answerable,
                declined=False,
            )
        )
    return samples


def score_samples(samples: list[EvalSample]) -> dict[str, float]:
    """Score `samples` with Ragas, judged by Gemini. Returns per-metric averages."""
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    from minddrill.config import get_settings

    settings_key = get_settings().gemini_api_key
    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=JUDGE_CHAT_MODEL, google_api_key=settings_key, temperature=0
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        # google_api_key is a real pydantic field (confirmed via model_fields);
        # Pylance's bundled stub for this class just doesn't declare it.
        GoogleGenerativeAIEmbeddings(
            model=JUDGE_EMBED_MODEL,
            google_api_key=settings_key,  # pyright: ignore[reportCallIssue]
        )
    )

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s.question,
                response=s.answer,
                retrieved_contexts=s.retrieved_contexts,
                reference=s.ground_truth,
            )
            for s in samples
        ]
    )

    # Small batches + generous retry/timeout: the free Gemini tier throttles
    # (429) under a burst of judge calls.
    run_config = RunConfig(max_workers=2, max_retries=5, max_wait=60)
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            LLMContextPrecisionWithReference(),
            ResponseRelevancy(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )
    return scores_from_result(result)


def scores_from_result(result) -> dict[str, float]:
    """Average a Ragas `EvaluationResult`'s per-row scores into our metric names.

    Split out from `score_samples` so the averaging/renaming logic is testable
    against a fake result (any object with `.to_pandas()`) without importing
    ragas or calling Gemini.
    """
    scores = result.to_pandas().mean(numeric_only=True)
    return {
        "faithfulness": float(scores.get("faithfulness", float("nan"))),
        "context_precision": float(
            scores.get("llm_context_precision_with_reference", float("nan"))
        ),
        "answer_relevancy": float(scores.get("answer_relevancy", float("nan"))),
    }


def gate(
    scores: dict[str, float], thresholds: dict[str, float]
) -> tuple[bool, list[str]]:
    """Pass iff every threshold key is present in `scores` and meets its floor."""
    failing = [
        metric
        for metric, floor in thresholds.items()
        if not (scores.get(metric, float("-inf")) >= floor)
    ]
    return not failing, failing


def write_report(
    out_dir: Path,
    scores_on: dict[str, float],
    scores_off: dict[str, float] | None,
    samples: list[EvalSample],
    passed: bool,
    failing: list[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": passed,
        "failing_metrics": failing,
        "scores_rerank_on": scores_on,
        "scores_rerank_off": scores_off,
        "sample_count": len(samples),
        "per_question": [
            {
                "question": s.question,
                "category": s.category,
                "answerable": s.answerable,
                "declined": s.declined,
            }
            for s in samples
        ],
    }
    json_path = out_dir / "eval_report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Eval report",
        "",
        f"**Gate: {'PASS' if passed else 'FAIL'}**"
        + (f" (failing: {', '.join(failing)})" if failing else ""),
        "",
        "## Scores (rerank on)",
        "",
    ]
    for metric, value in scores_on.items():
        lines.append(f"- {metric}: {value:.3f}")
    if scores_off is not None:
        lines += ["", "## Scores (rerank off — passthrough baseline)", ""]
        for metric, value in scores_off.items():
            lines.append(f"- {metric}: {value:.3f}")
        lines += ["", "## Rerank impact (on − off)", ""]
        for metric in scores_on:
            if metric in scores_off:
                delta = scores_on[metric] - scores_off[metric]
                lines.append(f"- {metric}: {delta:+.3f}")
    md_path = out_dir / "eval_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


async def _run(smoke: bool, out_dir: Path, compare_rerank: bool) -> int:
    from minddrill.config import get_settings
    from minddrill.db.testing import init_test_schema, resolve_test_database_url

    os.environ["DATABASE_URL"] = resolve_test_database_url()
    get_settings.cache_clear()
    await init_test_schema()

    from minddrill.db.session import SessionLocal
    from minddrill.providers.failover import get_providers
    from minddrill.rag.embedder import get_embedder
    from minddrill.rag.reranker import get_reranker

    items = select_subset(load_golden(GOLDEN_PATH), smoke)
    if not items:
        raise RuntimeError("no golden items selected — check the smoke subset filter")

    embedder = get_embedder()
    providers = get_providers()

    async with SessionLocal() as session:
        user_id = await ensure_eval_user(session)
        await session.commit()
        await ingest_golden(session, user_id, items, embedder)
        reranker_on = get_reranker()
        samples_on = await build_samples(
            session, user_id, items, embedder, providers, reranker_on
        )

        samples_off = None
        if compare_rerank:
            os.environ["RERANK_ENABLED"] = "false"
            get_settings.cache_clear()
            get_reranker.cache_clear()
            reranker_off = get_reranker()
            samples_off = await build_samples(
                session, user_id, items, embedder, providers, reranker_off
            )
            os.environ["RERANK_ENABLED"] = "true"
            get_settings.cache_clear()
            get_reranker.cache_clear()

    scores_on = score_samples(samples_on)
    scores_off = score_samples(samples_off) if samples_off is not None else None

    passed, failing = gate(
        scores_on,
        {"faithfulness": FAITHFULNESS_MIN, "context_precision": CONTEXT_PRECISION_MIN},
    )

    for metric, value in scores_on.items():
        print(f"{metric}: {value:.3f}")
    if not passed:
        print(f"GATE FAILED: {', '.join(failing)}")

    report_path = write_report(
        out_dir, scores_on, scores_off, samples_on, passed, failing
    )
    print(f"report written to {report_path}")

    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="run the small PR-gate subset only"
    )
    parser.add_argument(
        "--out", default="eval/results", help="directory to write the report to"
    )
    parser.add_argument(
        "--no-rerank-compare",
        action="store_true",
        help="skip the rerank on/off comparison run (faster, no artifact)",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(
        _run(args.smoke, Path(args.out), compare_rerank=not args.no_rerank_compare)
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
