# Slice 2 — Thin RAG path (vector only, synchronous)

Tables (alembic migration):
- documents(id, user_id fk, source_type, source_uri, title, content_hash,
  status, metadata jsonb, created_at). unique (user_id, content_hash).
- chunks(id, document_id fk, user_id, chunk_index, content, token_count,
  embedding vector(768), metadata jsonb, created_at).
  NOTE: hand-edit the migration to `CREATE EXTENSION IF NOT EXISTS vector;`
  and create the vector column + an HNSW index (alembic autogenerate won't).

Interfaces:
- Embedder (embed_texts, embed_query) -> Gemini text-embedding-004, dim 768.
- Load PDF with langchain PyPDFLoader. Split with RecursiveCharacterTextSplitter,
  ~600 tokens, 100 overlap.

Ingestion (SYNCHRONOUS this slice — one small PDF is fine):
- POST /ingest {source_type:"pdf", source_uri} for current_user.
  Skip if (user_id, content_hash) already exists (return existing document_id).
  Else: load -> split -> embed -> insert document + chunks (with user_id). 
  Return {document_id, chunk_count}.

Query (non-streaming this slice):
- POST /query {question} for current_user.
  Embed question -> vector search top-5 on chunks WHERE user_id = current_user
  -> build a prompt with the 5 chunks + numbered sources -> call Gemini
  (non-streaming) -> return {answer, sources:[{chunk_id, document_id}]}.

Tests:
- ingest a tiny fixture PDF -> chunks stored with correct user_id.
- query returns an answer and non-empty sources.
- ISOLATION: user B's query never returns user A's chunks. (Mock the LLM in
  unit tests; do one manual smoke test with the real Gemini key.)

What can go wrong: empty retrieval (return a graceful "no context" answer),
embedding dim mismatch, PDF parse failure, Gemini rate limit (429).