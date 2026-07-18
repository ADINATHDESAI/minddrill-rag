"""Short-term conversation memory: load recent turns, trim to a token budget.

This is deliberately separate from RAG retrieval — history comes only from the
`messages` table, never from the embeddings/BM25 store. `/chat` uses it to build
a prompt from the conversation itself, not from documents.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.models.message import Message

_CHARS_PER_TOKEN = 4  # same rough heuristic used when chunking documents

_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user using the conversation so far. "
    "Be concise and direct."
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


async def load_history(session: AsyncSession, session_id: uuid.UUID) -> list[Message]:
    result = await session.scalars(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at, Message.id)
    )
    return list(result)


def trim_history(messages: list[Message], budget: int) -> list[Message]:
    """Keep the most recent turns that fit `budget` tokens, dropping the oldest.

    The newest turn is always kept, even if it alone exceeds the budget.
    """
    kept: list[Message] = []
    total = 0
    for msg in reversed(messages):
        if kept and total + msg.token_count > budget:
            break
        kept.append(msg)
        total += msg.token_count
    kept.reverse()
    return kept


def build_chat_messages(history: list[Message]) -> list[dict]:
    messages = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}]
    messages.extend({"role": m.role, "content": m.content} for m in history)
    return messages
