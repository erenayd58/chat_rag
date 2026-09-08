# Database — PostgreSQL, migrations, and what is still a file

**PostgreSQL is authoritative for this application's relational state.** The
knowledge bases, the ingest ledger, the content identities and their analysis
state, the ingest journal and the gold set are rows. Nothing dual-writes and
nothing falls back to a file.

**The vector store is still authoritative for embeddings.** Chroma holds every
vector and every chunk, exactly as before, under the data directory. Moving it
to pgvector is the next step and is not started; the two stores are separate on
purpose until then.

```
                 FastAPI /api/v1        legacy Flask console
                        \                    /
                         \                  /
                      application / domain
                         /                  \
                        /                    \
              PostgreSQL                   vector store (Chroma)
   knowledge bases                         embeddings
   documents (the ingest ledger)           chunk rows
   content identity + analysis state       the lexical index
   ingest jobs
   gold set
```

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

Seven tables. `storage/models.py` carries the reasons; this is the shape.

| table | one row is | key |
|---|---|---|
| `knowledge_bases` | a named collection with its own chunker, embedding and vector store | `id` (the 8-character id `/api/v1` publishes) |
| `documents` | **one upload**: one file, ingested into one knowledge base | `id`; `doc_id` unique, and the identity every API addresses |
| `contents` | **the bytes**, and the analysis that belongs to them | `id`; `content_key` and `content_sha256` unique |
| `content_documents` | one upload's membership of one content, and the methods *it* selected | `(content_id, doc_id)`; `doc_id` unique on its own |
| `content_variants` | one chunking method run over one content | `(content_id, method)` |
| `ingest_jobs` | what a restart may answer a `job_id` with | `job_id` |
| `gold_set_entries` | one confirmed answer, per (knowledge base, question) | `entry_id` |

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

One path remains in the schema — `knowledge_bases.vector_db_path` — and it
addresses the *vector store*, which is still a directory until pgvector.

---

## What is still a file

Not everything durable belongs in a database, and these deliberately stay under
the data directory (`config/paths.py`):

| what | where | why |
|---|---|---|
| embeddings, chunk rows, the lexical index | `chroma/` | the vector store's job; pgvector replaces it next |
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
