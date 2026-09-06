# Configuration — where a setting comes from and what wins

For the developer who wants to change one number and not wonder which value
the process actually used.

## Precedence

```
  1. the real process environment      (a container, compose, CI, your shell)
  2. .env                              (applied once, by config/__init__.py)
  3. the application default           (the setting's own dataclass field)
```

**One exception, deliberate.** A *state path* — `VECTOR_DB_PATH`,
`STRUCTURED_PARSER_CACHE` — from `.env` is ignored once `CHAT_RAG_DATA_DIR`
is set. `.env` describes a developer's local layout (`VECTOR_DB_PATH=./chroma_db`
means "the store in my checkout"), and a deployment or a smoke run that
declared where its state lives must not have it moved back by a leftover file.
That is how a smoke run once opened the developer's real Chroma store. Rule 1
still applies: set the variable in the real environment and it wins, visibly.
Refusals are reported at start-up (`! VECTOR_DB_PATH=… ignored: …`).

`.env` is applied in `config/__init__.py`, before any config module reads
anything, so import order cannot change what a setting resolves to. It is the
only place in the application that reads a dotenv file, and a test asserts
that.

## The five owners

| owner | what it decides | validation |
|---|---|---|
| [config/paths.py](../config/paths.py) | where every piece of runtime state goes, under `CHAT_RAG_DATA_DIR` | resolves; reports refusals |
| [config/runtime.py](../config/runtime.py) | the server process: `FLASK_HOST`, `FLASK_PORT`, `WAITRESS_THREADS`, `WAITRESS_CHANNEL_TIMEOUT` | fail fast |
| [config/ingest.py](../config/ingest.py) | ingest workers, queue, deadlines, provider/embedding budgets, pipeline cache | fail fast |
| [config/query.py](../config/query.py) | query admission, answer budget, deadlines | fail fast |
| [config/settings.py](../config/settings.py) | everything else: models, endpoints, retrieval, chunking, parsing | fail fast on the strict ones |
| [utils/logger.py](../utils/logger.py) | `LOG_LEVEL`, `LOG_FILE_LEVEL`, rotation | **fail safe** — see below |

`Settings` builds the three limit objects at construction, so an invalid value
stops the process at start-up, when somebody is looking, rather than refusing
the first upload.

### Why logging validates differently

Every numeric limit refuses to start on a bad value. Logging does not: an
unrecognised level falls back to `INFO` and *says so* at start-up. The risky
direction is `DEBUG` — the file handler writes prompts, retrieved chunks and
answer context to a file that gets tailed, shipped and pasted into tickets — so
a typo must never be what turns that on, and a logging typo must not take the
service down either. That is the one place the categories diverge, and it is
deliberate. The fallback used to be silent; now it is reported.

## Defaults: one place each

A default is written on the dataclass field and nowhere else. The env readers
take the field, not a restated string:

```python
workers=_number(env, "INGEST_WORKERS", _DEFAULTS.workers, int)
```

`WAITRESS_THREADS` is the one every other ration is sized against
(`INGEST_SYNC_WAITERS` defaults to half of it, `QUERY_MAX_ACTIVE` to
`threads - sync_waiters - 1`). It has exactly one reader, `config/runtime.py`.
Before, `wsgi.py`, `config/ingest.py` and `config/query.py` each read it with
the default `8` written out, and `wsgi.py`'s parser silently forgave a bad
value while the others did not — so `WAITRESS_THREADS=-4` produced a server
with eight threads and limits sized against minus four.

## Cross-setting rules

Checked once, in `config.runtime.cross_check`, so no rule is stated twice:

| rule | outcome |
|---|---|
| `INGEST_SYNC_WAIT` < `WAITRESS_CHANNEL_TIMEOUT` | **refuses to start** — a synchronous upload would still be waiting when the server closes the connection it would answer on |
| `QUERY_MAX_ACTIVE + INGEST_SYNC_WAITERS` < `WAITRESS_THREADS` | **warns** — an explicit ration that leaves no free thread has always been honoured and named rather than overruled |
| `DEEP_ANALYSIS_CONCURRENCY` ≤ `PROVIDER_MAX_INFLIGHT` | **warns** — a job could never reach its own concurrency |

Warnings appear in the start-up banner as `! …` lines and in
`/api/ops/metrics` under `configuration.warnings`.

## Docker vs local

`.env.docker` is read by compose and is **not** the local `.env` (which points
at `localhost`, meaning the container itself). Every value there is either a
deliberate deployment override or a default restated for visibility, and
`tests/unit/test_configuration.py` requires each *difference* from the code to
be listed with a reason. The deliberate ones today:

| setting | container value | why |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | the container talks to Ollama on the host, not Azure |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | reaches the host from inside a container |
| `OLLAMA_MODEL` | `qwen2.5:3b` | the small model the demo image expects |
| `CHUNKER_TYPE` | `structure_first` | the frozen proof-of-concept selection |
| `RETRIEVAL_PROFILE` | `bm25_only` | needs no provider, so the image starts with no key |

Secrets go in `.env.docker.local`, which is git-ignored and overrides
`.env.docker`.

`env.example` works the same way: **commented-out lines show the application
default**; uncommented lines are the demo profile, which deliberately differs
because the defaults are the conservative no-provider ones. Both halves are
held to the code by the same test, so neither file can quietly become the
source of truth, and neither may document a setting no code reads (it used to
offer circuit breakers, retries, rate limits and a cache that nothing has ever
read).

## Secrets

Keys are configuration, never defaults or source constants. Most of this
application configures the *name* of the variable holding a key
(`ANSWER_API_KEY_ENV`, `EMBEDDING_API_KEY_ENV`, `DEEP_ANALYSIS_API_KEY_ENV`);
the key itself is read at request time and never stored, logged or serialised.
`FLASK_SECRET_KEY` falls back to a per-process random key with a warning.

`Settings.to_dict()` redacts everything named in `SECRET_ATTRIBUTES`, and
`Settings.effective_configuration()` — what the banner and `/api/ops/metrics`
print — carries no credential by construction. A test asserts a planted key
appears in neither.

## Diagnostics

`Settings.effective_configuration()` is the one answer to "what is this
instance running with". The start-up banner prints it and
`/api/ops/metrics` returns it under `configuration`; they cannot disagree.
It carries the data root, the resolved store and cache paths, the runtime /
ingest / query limits, the configured models and key-variable *names*, the
logging setup, and any warnings. It is not an environment dump.

## Adding a setting

1. Decide the owner from the table above.
2. Add the field to that module's dataclass **with its default** (or, for
   `Settings`, one `os.getenv` call), and read it with `_number` / `_text`.
3. Add its rule to that dataclass's `validate()`. If the rule involves another
   group, put it in `cross_check` instead — not in both.
4. Document it in `env.example`, commented out, showing the default. If the
   demo profile or the container needs a different value, add the line **and**
   its reason to `DELIBERATE_OVERRIDES` in
   `tests/unit/test_configuration.py`.
5. If an operator would need to see it while debugging, add it to
   `effective_configuration()` — never a credential.
