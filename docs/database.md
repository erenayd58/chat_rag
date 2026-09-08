# Database — PostgreSQL, migrations, and what is still a file

**PostgreSQL is authoritative for everything durable this application
keeps.** The knowledge bases, the ingest ledger, the content identities and
their analysis state, the ingest journal, the gold set — and, since Step 9,
the chunks and their embeddings — are rows. Nothing dual-writes and nothing
falls back to a file.

Chroma is gone. It is not an optional backend, not a fallback and not a
dependency; `pip install -r requirements.txt` no longer installs it, and the
one thing that still reads a Chroma store is the migration tool that empties
one (`tools/migrate_chroma_to_pgvector.py`, which imports it lazily and tells
you to install it).

```
                 FastAPI /api/v1        legacy Flask console
                        \                    /
                         \                  /
                      application / domain
                                 |
                            PostgreSQL
              knowledge bases
              documents (the ingest ledger)
              content identity + analysis state
              ingest jobs
              gold set
              vector_collections  — one corpus, and its embedding manifest
              chunk_vectors       — chunk text, metadata, embedding (pgvector)
```

The BM25 index is the one thing retrieval keeps outside the database, and it
always was: it is built in memory from the chunk rows on demand and never
persisted. What changed for it in Step 9 is only where the rows come from.

---

## Getting started

```bash
# 1. a database
docker compose -f docker-compose.test.yml up -d      # local development and the suite
#    or any PostgreSQL you like

# 2. tell the application where it is
cp env.example .env                                  # DATABASE_URL is in there

# 3. create the schema
alembic upgrade head

# 4. run
python -m wsgi
```

Step 3 is not optional and is not done for you. Nothing in the application
creates a table — not at import, not at start-up, not in a test fixture.
A fresh database is built by `alembic upgrade head` and by nothing else, which
is what makes the schema reviewable, repeatable and reversible.
`tests/storage/test_migrations.py` fails if any source file grows a
`create_all`.

With `DATABASE_URL` unset or unreachable, `python -m wsgi`, `python app.py` and
`python -m asgi` refuse to start and say which host they could not reach. There
is no degraded mode that serves without a database: every screen begins by
listing knowledge bases.

---

## Configuration

| variable | default | what it is |
|---|---|---|
| `DATABASE_URL` | **none — required** | `postgresql+psycopg://user:password@host:port/name`. A bare `postgresql://` is given the psycopg driver automatically. |
| `DATABASE_POOL_SIZE` | 5 | connections kept open per process |
| `DATABASE_MAX_OVERFLOW` | 5 | extra connections allowed under a burst |
| `DATABASE_POOL_TIMEOUT` | 30 | seconds a caller waits for one before failing |
| `DATABASE_POOL_RECYCLE` | 1800 | seconds after which an idle connection is replaced |
| `DATABASE_CONNECT_TIMEOUT` | 10 | seconds to wait for the TCP connection |
| `DATABASE_ECHO` | false | log every statement; development only, the log is not redacted |

`DATABASE_URL` has no application default and cannot have one: the only
fallback would be a credential written into a tracked file. Everything else
follows the ordinary precedence — real environment, then `.env`, then the
dataclass field in `config/database.py`. See
[configuration.md](configuration.md).

**Secrets.** `env.example` and `.env.docker` carry a development password for a
database that guards nothing. A deployment sets its own in the real
environment, or in `.env.docker.local`, which is not tracked. Nothing logs a
connection string with its credential in it: `config/database.py` sanitizes the
URL to `scheme://host:port/name` before it reaches the start-up banner or
`/api/ops/metrics`.

---

## The schema

Nine tables. `storage/models.py` carries the reasons; this is the shape.

| table | one row is | key |
|---|---|---|
| `knowledge_bases` | a named collection with its own chunker and embedding | `id` (the 8-character id `/api/v1` publishes) |
| `documents` | **one upload**: one file, ingested into one knowledge base | `id`; `doc_id` unique, and the identity every API addresses |
| `contents` | **the bytes**, and the analysis that belongs to them | `id`; `content_key` and `content_sha256` unique |
| `content_documents` | one upload's membership of one content, and the methods *it* selected | `(content_id, doc_id)`; `doc_id` unique on its own |
| `content_variants` | one chunking method run over one content | `(content_id, method)` |
| `ingest_jobs` | what a restart may answer a `job_id` with | `job_id` |
| `gold_set_entries` | one confirmed answer, per (knowledge base, question) | `entry_id` |
| `vector_collections` | **one searchable corpus**, and which embedding space it holds | `collection` — a knowledge base's id, or `VECTOR_DB_COLLECTION` for the one used when none is selected |
| `chunk_vectors` | one chunk: its text, its metadata and its embedding | `(collection, chunk_id)` |

### The two identities

A document and a content are **not** the same thing, and the schema keeps them
apart because collapsing them is the most expensive mistake available here:

* the same PDF uploaded into two knowledge bases is **two documents and one
  content** — one parse, one set of packaged variants, two ledger rows;
* what an upload *selected* lives on its membership row, not on the content, so
  an upload that asked for Standard does not suddenly display the Hybrid
  variant another upload of the same bytes happened to build;
* what the content *has* lives on the content, because a variant is built once
  and reused by every upload of it.

`visible = selected ∩ ready` is the rule the screens apply on top of that.

### The edges, and the two that are deliberately not foreign keys

Referential, with a real cascade:

* `content_documents.content_id → contents.id` — `ON DELETE CASCADE`
* `content_variants.content_id → contents.id` — `ON DELETE CASCADE`

Deleting a content takes its memberships and its variants, because nothing
points at any of them any more. Deleting *one membership* takes only that
upload's row and its choice; the content and its variants stay for the other
uploads. The content itself goes when the last membership does — a
reference-count rule the domain applies, in one transaction under the content's
row lock.

* `chunk_vectors.collection → vector_collections.collection` — `ON DELETE CASCADE`
* `vector_collections.kb_id → knowledge_bases.id` — `ON DELETE CASCADE`

Those last two are the pair that replaced "remove the directory, then remove
the record, and in that order". Deleting a knowledge base row removes its
collection and every vector in it, in the same transaction, by the database.
A vector that outlived its knowledge base would be an orphan a *later*
knowledge base could be matched against, which is exactly why this edge
cascades where the two below do not.

Not foreign keys, on purpose:

* `documents.kb_id`
* `contents.kb_id`

Deleting a knowledge base here deliberately leaves its documents' ledger rows
and their analyses behind, still naming the id that is gone; the console groups
them as *"Bilgi tabani silinmis kayitlar"* rather than dropping the record that
a file was ever ingested. `ON DELETE CASCADE` would silently lose that, and
`ON DELETE SET NULL` would lose the id the grouping is by. The Step 6
characterisation tests hold both behaviours
(`tests/migration/test_domain_relations.py`).

### File paths are not identity

The ingest ledger used to be a JSON object keyed by the **absolute path of the
uploaded file** — a staging file deleted the moment the job ended, on the
machine that ran the upload. A document is now a row: `documents.id` is its
database identity and `documents.doc_id` is the identity the product publishes.
`file_name` is kept because every screen shows it, and `source_path` is kept as
a diagnostic; neither is unique, indexed, or an address. No path has ever been
exposed through `/api/v1` and none is now.

No path remains in the schema. `knowledge_bases.vector_db_path` was the last
one, and `0002_pgvector_store` drops it: a knowledge base is identified by its
id, and its corpus is the `vector_collections` row whose `collection` is that
id. It was never exposed through `/api/v1` and there is now nothing to
expose.

---

## The vectors

### One column, several embedding spaces

`chunk_vectors.embedding` is declared `vector` with **no width**. That is a
decision, not an omission. This application supports more than one embedding
space at a time:

| where it comes from | width |
|---|---|
| `all-MiniLM-L6-v2` / `paraphrase-multilingual-MiniLM-L12-v2`, the local default | 384 |
| `qwen/qwen3-embedding-8b` through the gateway, the demo profile | 4096 |
| a lexical-only ingestion, which embeds nothing and stores a constant placeholder | 1 |

A knowledge base names its own embedding model, so two knowledge bases in one
database can legitimately be in two of those spaces at once. A `vector(n)`
column would have to pick one width and would reject every write from the
others, so the width is recorded per row (`embedding_dim`) instead, and the
invariant that matters is enforced where it belongs: **one collection holds one
width**. A write at a different width is refused by name; `replace_all` — a
re-index — is the one call allowed to change it, because it empties the
collection first.

`components/embedding/index_manifest.py` is the rule on top of that: a query is
never compared against vectors a different model produced. The manifest that
used to be an `embedding_index.json` beside the store is the manifest columns
on `vector_collections` now.

### Distance, and why nothing above it changed

Cosine, through pgvector's `<=>` operator, ordered ascending:

```
0.0  identical      1.0  orthogonal      2.0  opposite
```

That is the same number, on the same scale, in the same direction that Chroma
returned for a cosine collection. It has to be, because three things above the
store read it directly: the dense leg turns it into `1 - distance`, the console
turns it into `1 - distance / 2`, and the RRF fusion assumes the list arrives
nearest-first. `tests/migration/test_retrieval_parity.py` pins all of it —
ranking, ties, filtering, direction — against a second implementation.

### No ANN index, deliberately

There is no HNSW or IVFFlat index on `chunk_vectors`, and the search is an
exact scan of the collection. Two reasons, in order:

1. **pgvector cannot index this column.** Both index types require a column of
   fixed, declared width; `vector` with no width cannot carry either. Choosing
   a width to get an index back would break every knowledge base not in that
   space — the trade the previous section rejected.
2. **Correctness first.** An exact scan returns the true ranking. An ANN index
   returns an approximation, and every recall trade-off would have to be
   re-tuned against the RRF fusion and the gold set before it could be trusted.
   For corpora of this size — thousands of chunks per knowledge base, scoped by
   a btree index on `(collection, doc_id)` — the scan is not the bottleneck;
   the embedding call in front of it is.

When one production embedding width is fixed and a corpus is large enough to
need it, the change is a migration that adds a typed column and one index:

```sql
ALTER TABLE chunk_vectors ADD COLUMN embedding_4096 vector(4096);
CREATE INDEX ON chunk_vectors USING hnsw (embedding_4096 vector_cosine_ops);
```

HNSW rather than IVFFlat, because this workload writes continuously (every
ingest) and IVFFlat's lists have to be rebuilt as the data grows. Note that
`vector` tops out at 2000 dimensions for an index — a 4096-wide space needs
`halfvec` — which is one more reason it is not worth doing before a width is
actually fixed.

### Migrating an existing Chroma store

```bash
pip install chromadb                       # not a dependency any more
python -m tools.migrate_chroma_to_pgvector --root ./chroma_db --dry-run
python -m tools.migrate_chroma_to_pgvector --root ./chroma_db
```

It copies: chunk text byte for byte, metadata whole, and the vectors as the
floats Chroma already held — **no embedding is recomputed**, so no provider is
called and a migrated corpus answers with the ranking it had. The
`embedding_index.json` beside each store becomes the collection's manifest
row. It is idempotent (a chunk already there, unchanged, is skipped), scoped to
one knowledge base at a time, and reports migrated/skipped/failed counts. A
chunk already present with *different* content is reported as a conflict and
left alone unless `--overwrite` says otherwise.

If a store cannot give its embeddings back, the tool says so and writes nothing
for that batch rather than inventing a vector. The recovery is then the
controlled rebuild:
`POST /api/v1/knowledge-bases/<kb_id>/embedding-index/rebuild`, which re-embeds
the migrated text with the current model.

---

## What is still a file

Not everything durable belongs in a database, and these deliberately stay under
the data directory (`config/paths.py`):

| what | where | why |
|---|---|---|
| canonical units, packaged `chunks.jsonl`, Deep run trees, `viewer-payload.json` | `viewer-live/<content key>/` | large, regenerable from an ingest, read as files by the library's own reader; a row would only be a second name for a path |
| the parser's canonical-unit cache | `cache/canonical-units/` | a cache |
| embedding caches | `cache/embeddings/`, `cache/boundary-embeddings/` | caches |
| uploads waiting for their job | `uploads/` | transient; swept at start-up |
| logs | `logs/` | logs |

The **directory name** of an analysis is `contents.content_key`, so the
artifacts are addressed by a row rather than the row being the directory.

---

## Migrations

Alembic, configured in `alembic.ini`, scripts in `storage/migrations/`.

```bash
alembic upgrade head            # create or update the schema
alembic current                 # what this database is at
alembic heads                   # must print exactly one
alembic downgrade -1            # step back one revision
alembic revision --autogenerate -m "what changed"
```

The connection string is **not** in `alembic.ini`. `storage/migrations/env.py`
reads it through `config.database`, so a migration and the application can
never be pointed at two different databases — and no credential is in a tracked
file. A caller may still name one explicitly by setting `sqlalchemy.url` on a
`Config` object it passes in, which is how the fresh-install test builds a
throwaway database.

**One head.** `tests/storage/test_migrations.py` fails if the history branches,
because `alembic upgrade head` cannot choose between two of them and the place
to find that out is not a deployment window. It also compares the migrated
schema against the models and fails on any difference, so a column added to a
model without a migration cannot reach production.

After a migration is generated, read it. Autogenerate does not see a rename (it
sees a drop and an add), and it does not know that a `NOT NULL` column added to
a populated table needs a default or a backfill.

---

## Transactions and concurrency

The transaction boundary is the caller's, stated with a `with`:

```python
with session_scope() as session:
    KnowledgeBaseRepository(session).create(kb_id, config)
    DocumentRepository(session).upsert(record)
    # commits here, or rolls back everything on the way out
```

Operations that touch several records run in one of these, so a failure part
way through cannot leave a half-created domain graph — a knowledge base with a
document nobody can reach through it, or an analysis attached to an upload that
was never registered.

The invariants that used to rest on a process-local lock now rest on the
database, which means they hold for a second process too:

| invariant | what enforces it |
|---|---|
| one knowledge base per name | unique index on the normalised name; the loser gets the `ValueError` the routes already turn into a 400 |
| one row per upload | unique `documents.doc_id`, written as an upsert |
| one content per set of bytes | unique `contents.content_key`, created with `INSERT … ON CONFLICT DO NOTHING` and then locked |
| one membership per upload | unique `content_documents.doc_id` |
| competing writes to one analysis | `SELECT … FOR UPDATE` on the content row for the length of the change |
| two deletions of one knowledge base | the row is locked before its vector store is touched, so only one of them removes the directory |
| two deletions of the last upload of a content | the same lock; exactly one sees itself as last |

`tests/storage/test_concurrency.py` drives each of these with several threads
that start together.

No distributed lock was introduced and none is needed. The runtime is still one
process with one Viewer packaging worker (`wsgi.py` says why), and the
correctness above no longer depends on that being true.

---

## Connection lifecycle

One engine per process, built lazily on first use, with a pooled connection per
unit of work — never one engine per repository call. `storage/engine.py` owns
it:

* `pool_pre_ping` so a connection the server closed overnight is replaced
  rather than handed to a request;
* `session_scope()` closes every session, which is what returns its connection
  to the pool; a session left to the garbage collector is how a pool leaks;
* `dispose()` on shutdown, after the ingest workers have drained — their last
  act is the ledger write, and that is the one write a job must not lose;
* `DATABASE_CONNECT_TIMEOUT` so an unreachable database is a start-up message,
  not a process that hangs before it binds.

Importing a module never opens a connection. `python -m cli manifest`,
`tools/import_smoke.py` and every test that touches no record all run on a
machine with no PostgreSQL at all.

---

## Health

`/api/v1/health` and `/api/health` are unchanged: same fields, same status
codes, same meaning. The database shows up in them the way every other
dependency does — through `state` and `reasons`. A database that cannot be
reached makes `services.kb_manager.list()` raise, which is already reported as
`degraded` with a redacted reason.

`/api/ops/metrics` carries the detail: whether the database is configured and
reachable, the sanitized URL, and how much of the connection pool is checked
out. No credential reaches either body.

---

## Importing an existing installation

A machine that has been running this console before Step 8 has five files of
state. They are gitignored, so no clone carries them, and there is nothing in
this repository to migrate — which is why this ships as an explicit tool rather
than as data inside a migration.

```bash
python tools/import_legacy_state.py --dry-run    # read and validate, write nothing
python tools/import_legacy_state.py
```

It reads `.knowledge_bases.json`, `.ingested_documents.json`, `.gold_set.json`,
`.ingest-jobs/*.json` and `viewer-live/*/state.json` — through the same
`CHAT_RAG_DATA_DIR`-aware resolution that wrote them — and writes them into the
database.

* **Idempotent.** Every write is an upsert on the record's own identity, so
  running it twice is running it once.
* **It never touches the files.** They are left exactly where they are, so a
  failed import can be run again and the files kept until you are satisfied.
  Deleting them is a decision, not a step.
* **Relationships are reported, not enforced.** A document naming a knowledge
  base that is gone is normal here and is counted, not rejected; a hundred of
  them means something else went wrong.
* **A file it cannot read stops that file and nothing else**, is named, and
  makes the exit status non-zero.

After it has run, PostgreSQL is authoritative. There is no dual-write mode and
nothing reads those files again.

---

## Testing

The suite needs a PostgreSQL of its own — never a developer's:

```bash
docker compose -f docker-compose.test.yml up -d
python -m pytest
```

The schema is built once per session by `alembic upgrade head` (the same
migrations a deployment runs) and every table is truncated before each test, so
a test that used to isolate itself by naming its own JSON file now gets an empty
database instead. `CHAT_RAG_TEST_DATABASE_URL` points the suite elsewhere.
[testing.md](testing.md) has the rest.

---

## Where to go next

* [architecture.md](architecture.md) — what the system is and which layer owns what
* [configuration.md](configuration.md) — every setting, and what wins
* [operations.md](operations.md) — running it, and what each refusal means
* `storage/models.py` — the schema, and the reasoning behind each edge
