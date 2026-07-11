"""chunks bm25 index

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-11

Hand-edited: alembic autogenerate does not emit the pg_search extension or a
`USING bm25` index — ParadeDB's index type is invisible to SQLAlchemy's
reflection, so the DDL is written explicitly below.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    # Including user_id lets the per-user isolation filter push into the index.
    op.execute(
        "CREATE INDEX chunks_bm25 ON chunks "
        "USING bm25 (id, content, user_id) WITH (key_field='id')"
    )


def downgrade() -> None:
    op.drop_index("chunks_bm25", table_name="chunks")
