# Changelog

What changed in the **published** API of `chat-rag`, release by release. The
policy this follows — what is public, what a bump means, how a removal is
announced — is [docs/library-api.md](docs/library-api.md).

Only the library surface is recorded here. The product's own history is the
git log; a change to `/api/v1` is recorded in [docs/api-v1.md](docs/api-v1.md).

This project follows [Semantic Versioning](https://semver.org), at 0.x: the
surface is published and its changes are recorded, and a minor bump may still
change it.

## [Unreleased]

### Fixed

- `Engine.migrate()` and `migrate_database()` report a failed migration as
  that migration's own error, and release the advisory lock. The unlock used
  to run inside the aborted transaction, so what came back was
  `InFailedSqlTransaction`, and the lock stayed with a pooled connection --
  every other process's migrate then waited on it for as long as this one
  lived.

## [0.1.0]

The first published surface. `chat_rag.api` became the Python API over the
engine that was already there, and this release is where it stopped being an
internal arrangement and became a promise.

### Added

- **`Engine`** — one configured engine, as a context manager. Owns its own
  container, its own runtime (connection pool, provider budgets, metrics,
  packaging queue, analysis engine), its own session id and, given
  `EngineConfig(data_dir=...)`, its own files. Two in one process share
  nothing but the database.
- **`EngineConfig`** — the stated subset of `Settings`. A field left unset
  means *this caller did not say*, and is then read the way the server reads
  it. `open_engine(**settings)` is the same thing from keywords.
- **`Settings`** — published, because `EngineConfig` is openly a subset of it
  and `EngineConfig.build()` returns one.
- **`KnowledgeBases`, `KnowledgeBase`, `Document`, `IngestJob`, `Analysis`** —
  the handles: create a knowledge base, ingest a document from a path, bytes
  or an open file, build and extend its chunking analysis, compare chunkings,
  search, ask.
- **`Answer`, `Source`, `Hit`, `Chunk`, `Comparison`, `Arm`, `Health`,
  `Method`** — frozen result types, projected from the use cases' own
  dictionaries independently of the `/api/v1` schemas, so a Python rename is
  not an HTTP break.
- **The refusals**, re-exported rather than wrapped: `ApplicationError`,
  `InvalidRequest`, `NotFound`, `Conflict`, `Unavailable`, `NotReady`,
  `ProcessingFailed`, and the resource-control four — `IngestOverloaded`,
  `IngestInterrupted`, `QueryOverloaded`, `QueryTimeout`.
- **`Engine.migrate()`** and **`migrate_database()`**, answering with a
  **`Migration`** — the schema, created or brought to head from the
  migrations inside the installed package, under an advisory lock, without
  `alembic.ini` or a checkout. `outcome` is `created`, `upgraded` or
  `current`; an unreachable database is `Unavailable`, a failed migration
  `ProcessingFailed`.
- **`chat_rag.__version__`**, read from the installed distribution's metadata.
- **`py.typed`**, so the annotations on all of the above are visible to a
  consumer's type checker.
- **Optional dependencies**: `local` (sentence-transformers, ollama) and `pdf`
  (pymupdf, pymupdf4llm, python-docx), with `all` for both. Installing neither
  leaves an engine that imports, serves and answers through a gateway; asking
  it for what an extra provides is refused by name.

### Fixed

- The Alembic migrations now ship in the wheel. `migrations/` and
  `migrations/versions/` have no `__init__.py` — Alembic loads them by path —
  so `packages.find` never saw them, and an installed library could not create
  its own schema.
- The wheel names where `amsc-poc` comes from: a direct reference to the
  pinned commit, so `pip install <wheel>` and `pip install <sdist>` resolve
  with nothing else named. Both are installed into an empty environment and
  used, first migration included, by `tests/integration/test_clean_install.py`.
- A knowledge base's pipeline is built from the engine's own settings. It
  used to re-read the environment, so `EngineConfig(retrieval_profile=...,
  embedding_provider=..., read_environment=False)` reached the default
  pipeline and none of the ones that ingest and answer.
- An ingest no longer fails because a progress line could not be printed.
  The pipeline narrates to stdout, and a program whose stdout is on a legacy
  code page (a redirected stream on Windows) got `UnicodeEncodeError` inside
  the ingest; the line is now escaped instead. The product's entrypoints
  already reconfigured their streams and never saw it.

### Notes

- **PostgreSQL is required.** It is not an extra and there is no mode without
  it.
- **Neither `chat-rag` nor `amsc-poc` is on an index.** The wheel installs
  by itself, from a file or from the repository, but the installing machine
  needs `git`; and a wheel carrying a direct reference cannot go to PyPI.
- Importing `chat_rag` still costs nothing: the published names resolve on
  first use, so reading `chat_rag.__version__` does not load the pipeline.

[Unreleased]: https://github.com/erenayd58/chat_rag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/erenayd58/chat_rag/releases/tag/v0.1.0
