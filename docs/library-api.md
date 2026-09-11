# The library API — what is public, and what that promises

`chat-rag` is a distribution as well as a product. This says what a consumer
may depend on, what may change under them, and how they will be told.

The short version: **the public API is `chat_rag.api.__all__` and nothing
else.** Everything under it is free to change without notice.

---

## The surface

```python
from chat_rag import Engine, EngineConfig

with Engine(EngineConfig(database_url="postgresql+psycopg://...")) as engine:
    engine.migrate()                            # the schema, created or brought to head
    kb = engine.knowledge_bases.create("Reports")
    document = kb.ingest("report.pdf")
    document.analysis().request()
    hits = kb.search("liquidity")
    answer = kb.ask("What changed?")
```

Thirty names, in four groups. `from chat_rag import X` and
`from chat_rag.api import X` are the same object; there is one list.

| group | names |
|---|---|
| the engine and its configuration | `Engine`, `EngineConfig`, `Settings`, `open_engine`, `migrate_database` |
| what it holds | `KnowledgeBase`, `KnowledgeBases`, `Document`, `IngestJob`, `Analysis` |
| what its calls answer with | `Answer`, `Arm`, `Chunk`, `Comparison`, `Health`, `Hit`, `Method`, `Migration`, `Source` |
| what its calls refuse with | `ApplicationError`, `InvalidRequest`, `NotFound`, `Conflict`, `Unavailable`, `NotReady`, `ProcessingFailed`, `IngestOverloaded`, `IngestInterrupted`, `QueryOverloaded`, `QueryTimeout` |

The refusals are the application's own classes, re-exported — not a
facade-shaped copy. Catching `chat_rag.NotFound` catches what the engine
raises, whichever surface it was reached through.

`Settings` is published because `EngineConfig` is openly *a stated subset of
it*: `EngineConfig.build()` returns one, and a caller who needs a field
`EngineConfig` does not name has to be able to construct one. It is a value
object — constructing one reads nothing — and its fields are documented in
[configuration.md](configuration.md). The frozen value objects it holds
(`IngestLimits`, `QueryLimits`, `DatabaseSettings`, `PathSettings`,
`RuntimeLimits`, in `chat_rag.config`) are covered by the same policy, because
you cannot construct a `Settings` without them.

### What is not public

Everything else in the package: `chat_rag.application`, `chat_rag.components`,
`chat_rag.pipeline`, `chat_rag.storage`, `chat_rag.runtime`, `chat_rag.core`,
`chat_rag.utils`. They are importable — this is Python — and they change
without notice.

Two are worth naming because they are *reachable from* the public API and are
still not part of it:

* **`Engine.services`** returns the `Services` container. It is the escape
  hatch, deliberately: a use case the facade does not wrap is reached through
  it. Using it means taking on what the facade does for you — passing the
  session id, and running inside `engine.activate()` — and it means tracking
  changes to `chat_rag.application`.
* **`Answer.diagnostics`, `Analysis.state`, `Health.raw`, `Source.raw`, and
  the mappings returned by `Engine.metrics()`, `Document.units()`,
  `Analysis.payload()` and `Analysis.chunks()`** are pass-through: they carry
  what the pipeline, the packager or the store produced. Their *presence* is
  public; their *contents* are not. Pinning them would freeze the internals
  the facade exists to keep free.

`tests/unit/test_public_surface.py` enforces all of this. It checks that the
exported names are exactly the published list, that both import paths agree,
that no internal type appears in a public signature — reading resolved
annotations, not the source text — and that no public **parameter** demands an
internal type at all. Returns may be escape hatches; a parameter that requires
a `Services` is a method only this repository can call.

---

## Versioning

`chat-rag` is at **0.x**, and 0.x means what it usually means: the surface is
published and its changes are recorded, and a **minor** bump may still break
it. `pip install "chat-rag~=0.1.0"` if that matters to you.

| bump | means |
|---|---|
| patch (`0.1.0` → `0.1.1`) | behaviour fixes. Nothing published is removed or renamed |
| minor (`0.1.0` → `0.2.0`) | new names, and possibly removals — each one announced in [CHANGELOG.md](../CHANGELOG.md) and deprecated first where it can be |
| major (`0.x` → `1.0.0`) | the surface stops changing in minors. Not yet — see the limitations below |

The version lives in `pyproject.toml` and nowhere else; `chat_rag.__version__`
reads the installed distribution's metadata.

## Deprecation

One removal, three steps, and never fewer than one minor release between the
first and the last:

1. **Announce.** The name keeps working, gains a `DeprecationWarning` naming
   what to use instead, and gets a `### Deprecated` entry in the changelog
   saying which release will remove it.
2. **Wait.** At least one minor release in which it warns and still works.
3. **Remove**, in a minor bump, with a `### Removed` entry.

A name that can be replaced in place — a rename, a widened argument — keeps an
alias through step 2 rather than making callers change twice.

Two things are exempt, and both are exempt because they were never promised:
anything outside `chat_rag.api.__all__`, and the *contents* of the
pass-through mappings listed above.

---

## The schema

The migrations ship in the wheel, and applying them is a call on the surface
rather than a step outside it:

```python
from chat_rag import Engine, EngineConfig, migrate_database

migrate_database(database_url="postgresql+psycopg://...")   # a setup script, a deploy step

with Engine(EngineConfig(database_url="postgresql+psycopg://...")) as engine:
    done = engine.migrate()                                  # or the engine's own
    done.outcome                                             # 'created' | 'upgraded' | 'current'
```

Both are idempotent and safe at every start: an empty database gets the
schema, one that is behind is brought to head, one already at head reports
`current` and applies nothing. The upgrade runs under a PostgreSQL advisory
lock, so two programs starting together cannot both migrate. No `alembic.ini`
and no checkout is involved — the migrations are found from where the package
is. A database that is not configured or cannot be reached is refused with
`Unavailable`; a migration that was attempted and failed (a server without
the `vector` extension is the usual one) with `ProcessingFailed`, cause
chained. `migrate_database` is `Engine.migrate` on an engine built for the
call and closed after it.

---

## Installing

```bash
pip install chat-rag              # the engine
pip install chat-rag[local]       # + sentence-transformers, ollama
pip install chat-rag[pdf]         # + pymupdf, pymupdf4llm, python-docx
pip install chat-rag[all]         # both
```

`chat-rag` is not on PyPI. Install it from the built artifact (`python -m
build`, then `pip install dist/chat_rag-*.whl`) or straight from the
repository (`pip install "chat-rag @ git+https://github.com/erenayd58/chat_rag.git"`).
Either way nothing else has to be named: the chunking library, `amsc-poc`,
is on no index either, and the wheel's metadata carries a direct reference to
the commit it is pinned to — the same commit `requirements.txt` names, held
equal by `tests/unit/test_amsc_pin.py` — so pip fetches it on its own. The
installing machine needs `git` for that, as the Docker build does. A wheel
with a direct reference in it cannot be uploaded to PyPI; that is a fact
about this distribution's current state, not a plan.

**PostgreSQL is required, not an extra.** The knowledge bases, the ingest
ledger, the content identities and their analysis state, the ingest journal,
the gold set and the chunk vectors are all rows. There is no degraded mode
that runs without one, so SQLAlchemy, psycopg, Alembic and pgvector are core
dependencies and the schema (`chat_rag/storage/migrations/`) ships in the
wheel — see [The schema](#the-schema) above for applying it.

The extras are the two things a deployment may genuinely not need:

| extra | what it adds | what you lose without it |
|---|---|---|
| `local` | `sentence-transformers` (and torch), the `ollama` client | local embedding models and local answers. A gateway provider (`EMBEDDING_PROVIDER=openrouter`, `ANSWER_PROVIDER=openrouter`) and `RETRIEVAL_PROFILE=bm25_only` need none of it |
| `pdf` | `pymupdf`, `pymupdf4llm[layout]`, `python-docx` | ingesting anything but text and Markdown |

Both degrade by refusing, never by pretending. Without `[local]`, asking for a
local model raises `ConfigurationException` with the install line in the
message; without `[pdf]`, `ParserFactory` registers no PDF parser and a PDF is
refused while a `.md` file ingests exactly as before.

`tools/wheel_smoke.py` builds the wheel (and, with `--sdist`, the source
distribution), installs each into an empty interpreter naming nothing else,
and checks every claim above — including that importing the surface pulls in
no torch and, given `--database-url`, that the first use works: migrate,
create a knowledge base, ingest a Markdown document, search it.
`tests/integration/test_clean_install.py` runs the same functions from the
suite against a throwaway database, for both artifacts.

---

## Limitations a consumer should know about

These are properties of the library today, not bugs to be surprised by.

* **Neither `chat-rag` nor `amsc-poc` is on an index.** The wheel installs
  by itself — its metadata says where `amsc-poc` comes from — but the
  installing machine needs `git` and a network, and the artifact cannot be
  published to PyPI while it carries a direct reference.
* **One process, one engine's worth of state.** The Viewer packager is a
  thread over an in-memory queue, the pipeline cache holds built pipelines,
  and the provider budgets are semaphores. Two `Engine`s in one process are
  properly separate ([architecture.md](architecture.md)); two *processes*
  share nothing but the database, so the packaging queue and the budgets do
  not span them. See [limitations.md](limitations.md).
* **A data root separates files, not rows.** `EngineConfig(data_dir=...)`
  gives an engine its own packaged analyses, staged uploads and caches. Two
  engines under one `DATABASE_URL` still share every record.
* **Applying the schema is still the consumer's decision.** `Engine.migrate()`
  and `migrate_database()` do it; nothing does it for you, because a library
  that alters a database on construction has spoken for a program that has
  not. Downgrades and new revisions are Alembic operations on a checkout.
* **`configure_logging` is not called for you.** A library installs no
  handlers; `chat_rag` attaches a `NullHandler` and nothing else.
* **Thread defaults are the process's.** Call
  `chat_rag.process.apply_thread_defaults()` before the first import of
  `chat_rag.api` if you will load more than one local model in one process.
  It is not public API; it is a documented process decision.
