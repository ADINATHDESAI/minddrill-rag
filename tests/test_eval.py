"""Tests for the offline eval runner (eval/run_eval.py).

No real Gemini/Ragas network calls: `scores_from_result` is tested against a
fake result object, and `build_samples` uses the same fakes (FakeEmbedder,
FakeLLM, FakeReranker) the rest of the suite uses. The DB guard is exercised
directly to confirm the eval can never resolve to a non-`_test` database.
"""

import uuid

import pandas as pd
import pytest

from minddrill.db.testing import resolve_test_database_url
from minddrill.rag.ingest import ingest_pdf
from minddrill.rag.schemas import IngestRequest

from eval.run_eval import (
    GoldenItem,
    build_samples,
    ensure_eval_user,
    gate,
    load_golden,
    scores_from_result,
    select_subset,
)


class _FakeRagasResult:
    """Stands in for a Ragas `EvaluationResult`: only `.to_pandas()` is used."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame


def test_scores_from_result_averages_tiny_fixture_set():
    # Three rows, as if scoring a tiny 3-item fixture set.
    frame = pd.DataFrame(
        {
            "user_input": ["q1", "q2", "q3"],
            "faithfulness": [1.0, 0.8, 0.9],
            "llm_context_precision_with_reference": [0.9, 0.7, 1.0],
            "answer_relevancy": [0.95, 0.85, 0.9],
        }
    )

    scores = scores_from_result(_FakeRagasResult(frame))

    assert scores["faithfulness"] == pytest.approx(0.9)
    assert scores["context_precision"] == pytest.approx(0.8666666, rel=1e-4)
    assert scores["answer_relevancy"] == pytest.approx(0.9)


def test_gate_passes_when_all_thresholds_met():
    scores = {"faithfulness": 0.9, "context_precision": 0.75}
    thresholds = {"faithfulness": 0.85, "context_precision": 0.70}

    passed, failing = gate(scores, thresholds)

    assert passed is True
    assert failing == []


def test_gate_fails_on_metric_below_threshold():
    # A deliberately bad faithfulness score should fail the gate non-zero.
    scores = {"faithfulness": 0.40, "context_precision": 0.90}
    thresholds = {"faithfulness": 0.85, "context_precision": 0.70}

    passed, failing = gate(scores, thresholds)

    assert passed is False
    assert failing == ["faithfulness"]


def test_gate_fails_when_a_metric_is_missing():
    passed, failing = gate({"faithfulness": 0.9}, {"context_precision": 0.70})

    assert passed is False
    assert failing == ["context_precision"]


_TINY_MD = (
    "## Vacation\n\nFull-time employees receive 20 days of PTO per year.\n\n"
    "## Security\n\nMFA is required for all internal systems.\n"
)


async def test_build_samples_on_tiny_fixture_set(
    db_session, embedder, reranker, llm, tmp_path
):
    doc_path = tmp_path / "tiny_handbook.md"
    doc_path.write_text(_TINY_MD, encoding="utf-8")
    user_id = await ensure_eval_user(db_session)
    await db_session.commit()
    await ingest_pdf(
        db_session,
        user_id,
        IngestRequest(source_type="markdown", source_uri=str(doc_path)),
        embedder,
    )

    items = [
        GoldenItem(
            question="How many PTO days do full-time employees get per year?",
            ground_truth="20 days.",
            source_doc=str(doc_path),
            category="numeric",
            answerable=True,
        ),
        GoldenItem(
            question="Is MFA required?",
            ground_truth="Yes.",
            source_doc=str(doc_path),
            category="factoid",
            answerable=True,
        ),
    ]

    samples = await build_samples(db_session, user_id, items, embedder, [llm], reranker)

    assert len(samples) == 2
    for sample, item in zip(samples, items):
        assert sample.declined is False
        assert sample.ground_truth == item.ground_truth
        assert sample.retrieved_contexts  # non-empty for an answerable question
        assert sample.answer  # the fake provider's canned tokens, joined


async def test_build_samples_declines_when_retrieval_finds_nothing(
    db_session, embedder, reranker, llm
):
    # A user with no ingested documents at all: retrieval is empty, so the
    # pipeline must decline rather than fabricate an answer.
    isolated_user = uuid.uuid4()
    items = [
        GoldenItem(
            question="Does the company offer pet insurance?",
            ground_truth="The handbook does not mention pet insurance.",
            source_doc="unused.md",
            category="unanswerable",
            answerable=False,
        )
    ]

    samples = await build_samples(
        db_session, isolated_user, items, embedder, [llm], reranker
    )

    assert len(samples) == 1
    assert samples[0].declined is True
    assert samples[0].retrieved_contexts  # placeholder context, never empty


def test_load_golden_parses_the_real_golden_set():
    items = load_golden("eval/golden/golden.jsonl")

    assert len(items) > 30
    assert any(not item.answerable for item in items)
    assert all(item.question and item.ground_truth for item in items)


def test_select_subset_smoke_includes_an_unanswerable_item():
    items = load_golden("eval/golden/golden.jsonl")

    smoke = select_subset(items, smoke=True)

    assert 0 < len(smoke) < len(items)
    assert any(not item.answerable for item in smoke)


def test_select_subset_full_returns_everything():
    items = load_golden("eval/golden/golden.jsonl")

    assert select_subset(items, smoke=False) == items


def test_resolve_test_database_url_never_targets_the_dev_db(monkeypatch):
    from minddrill.config import get_settings

    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/minddrill"
    )
    get_settings.cache_clear()
    try:
        resolved = resolve_test_database_url()
    finally:
        get_settings.cache_clear()

    assert resolved.endswith("minddrill_test")
    assert resolved != "postgresql+asyncpg://u:p@localhost:5432/minddrill"


def test_resolve_test_database_url_aborts_without_a_db_name(monkeypatch):
    from minddrill.config import get_settings

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/")
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="no database name"):
            resolve_test_database_url()
    finally:
        get_settings.cache_clear()
