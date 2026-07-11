"""Hybrid retrieval: pgvector arm + pg_search BM25 arm, fused with RRF.

Chunks are inserted directly with hand-crafted embeddings and content so each
arm can be exercised in isolation, independent of the fake embedder's hashing.
"""

import uuid

import pytest

from minddrill.config import get_settings
from minddrill.models.chunk import Chunk
from minddrill.models.document import Document
from minddrill.models.user import User
from minddrill.rag.retrieve import Retriever, _rrf

_DIM = get_settings().embed_dim


def _unit(*dims: int) -> list[float]:
    """A vector with 1.0 at each given dimension (others 0)."""
    vec = [0.0] * _DIM
    for d in dims:
        vec[d] = 1.0
    return vec


async def _seed_user(session) -> uuid.UUID:
    user = User(username=f"u_{uuid.uuid4().hex[:8]}", password_hash="x")
    session.add(user)
    await session.flush()
    doc = Document(
        user_id=user.id,
        source_type="pdf",
        source_uri="mem://test",
        content_hash=uuid.uuid4().hex,
        status="done",
    )
    session.add(doc)
    await session.flush()
    session.__dict__["_doc_id"] = doc.id  # stash for _add_chunk
    return user.id


async def _add_chunk(session, user_id, content: str, embedding: list[float]) -> Chunk:
    chunk = Chunk(
        document_id=session.__dict__["_doc_id"],
        user_id=user_id,
        chunk_index=0,
        content=content,
        token_count=len(content.split()),
        embedding=embedding,
    )
    session.add(chunk)
    await session.flush()
    return chunk


async def test_bm25_arm_finds_keyword_chunk_vector_ranks_low(db_session):
    """The keyword-exact chunk is returned by BM25 even though vector ranks it last."""
    user_id = await _seed_user(db_session)
    query_vec = _unit(0)
    # Keyword match, but embedding orthogonal to the query (vector ranks it last).
    kw = await _add_chunk(
        db_session, user_id, "photosynthesis converts light to sugar", _unit(1)
    )
    # No keyword, but embedding aligned with the query (vector ranks it first).
    other = await _add_chunk(
        db_session, user_id, "the harbor was quiet at dawn", _unit(0)
    )

    bm25 = await Retriever(db_session).bm25_search("photosynthesis", user_id, k=20)
    assert kw.id in bm25
    assert other.id not in bm25

    semantic = await Retriever(db_session).semantic_search(query_vec, user_id, k=20)
    # The keyword chunk the BM25 arm surfaced is the one vector search ranks last.
    assert semantic[0] == other.id
    assert semantic[-1] == kw.id


async def test_semantic_arm_finds_close_chunk_without_keyword_overlap(db_session):
    """The semantically-close chunk wins the vector arm despite zero word overlap."""
    user_id = await _seed_user(db_session)
    query_vec = _unit(0)
    # No overlap with the query terms, but embedding aligned with the query.
    close = await _add_chunk(
        db_session, user_id, "the harbor was quiet at dawn", _unit(0)
    )
    # Shares query words, but embedding far from the query.
    far = await _add_chunk(
        db_session, user_id, "quantum entanglement physics", _unit(1)
    )

    semantic = await Retriever(db_session).semantic_search(query_vec, user_id, k=20)
    assert semantic[0] == close.id
    assert semantic[-1] == far.id


async def test_rrf_ranks_both_arm_chunk_first(db_session):
    """A chunk ranked high by BOTH arms beats chunks ranked high by only one."""
    user_id = await _seed_user(db_session)
    query_vec = _unit(0)
    both = await _add_chunk(db_session, user_id, "keyword alpha", _unit(0))
    vec_only = await _add_chunk(db_session, user_id, "gamma delta", _unit(0, 1))
    bm_only = await _add_chunk(db_session, user_id, "keyword beta", _unit(2))

    fused = await Retriever(db_session).hybrid_search(
        "keyword", query_vec, user_id, k=5
    )
    ids = [c.id for c in fused]
    assert ids[0] == both.id
    assert {vec_only.id, bm_only.id} <= set(ids)


def test_rrf_math_hand_checked():
    """RRF fuses by rank with k=60: score(id) = sum 1/(60 + rank), rank 1-based."""
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    # a,b,c and b,c,a -> b appears at ranks (2,1), a at (1,3), c at (3,2).
    fused = _rrf([[a, b, c], [b, c, a]], k=60)
    # b: 1/62 + 1/61 = 0.032522; a: 1/61 + 1/63 = 0.032266; c: 1/63 + 1/62 = 0.032002
    assert fused == [b, a, c]


async def test_both_arms_isolate_by_user(db_session):
    """Neither arm returns another user's chunks — the real multi-tenant boundary."""
    a_id = await _seed_user(db_session)
    a_chunk = await _add_chunk(db_session, a_id, "launch code alpha seven", _unit(0))

    b_id = await _seed_user(db_session)
    # B's chunk shares the keyword and embedding, so a leak would be unmistakable.
    b_chunk = await _add_chunk(db_session, b_id, "launch code beta nine", _unit(0))

    query_vec = _unit(0)
    semantic = await Retriever(db_session).semantic_search(query_vec, b_id, k=20)
    assert a_chunk.id not in semantic
    assert semantic == [b_chunk.id]

    bm25 = await Retriever(db_session).bm25_search("launch code", b_id, k=20)
    assert a_chunk.id not in bm25
    assert bm25 == [b_chunk.id]
