# Testing and validation — what to run, and in what order

Three layers, and they answer different questions:

| layer | question it answers | cost |
|---|---|---|
| the suites | does the code do what it claims? | seconds to minutes |
| the smokes | can this build start at all? | seconds |
| the reproducibility gate | can *another machine* install and run the declared source? | minutes |

A green suite is not a working build — this project has the scar to prove it,
and that is why the third layer exists.

No test calls a real provider. `tests/conftest.py` moves the whole session out
of the checkout into a temporary directory and blanks the provider keys before
any application module is imported, so a test cannot reach a live endpoint or
touch your knowledge bases even by accident.

---

## The suites

Run both from their own repository root.

The `chat_rag` suite needs a PostgreSQL of its own, because that is where the
application's records live. Start it once and leave it running:

```bash
docker compose -f docker-compose.test.yml up -d
```

It listens on port 55432 so it cannot collide with a PostgreSQL already
installed on the machine, its credentials guard nothing, and it keeps its data
in memory — the schema is rebuilt from the migrations at the start of every
session anyway. `CHAT_RAG_TEST_DATABASE_URL` points the suite somewhere else;
without a database the session refuses to run and says so, rather than falling
back to something that would pass for the wrong reason.

Two things follow, and both are deliberate:

* **the schema under test is the deployed one.** `tests/conftest.py` drops the
  schema and runs `alembic upgrade head` — the same migrations a deployment
  runs. No fixture calls `create_all`, so a model that has outgrown its
  migration fails here rather than in production.
* **isolation is an empty database, not a temporary file.** Every table is
  truncated before every test. A test that used to isolate itself by pointing a
  store at its own `tmp_path` still passes that path — the stores still accept
  it — and gets a fresh database instead.

```bash
# chat_rag (this repo) — the venv's interpreter, from the repo root
python -m pytest -q                       # everything
python -m pytest tests/storage -q         # the repositories, invariants, transactions, concurrency, Alembic
python -m pytest tests/unit -q            # fast: no server, no store
python -m pytest tests/application -q     # the product's behaviour, with no framework at all
python -m pytest tests/integration -q     # the real app through its test client
python -m pytest tests/unit/test_query_limits.py -q          # one file
python -m pytest -q -k "ingest and restart"                  # by name
```

```bash
# chunk — Python 3.11 is what the library is developed against
py -3.11 -m pytest                        # everything
py -3.11 -m pytest tests/unit             # fast
py -3.11 -m pytest tests/integration      # frozen-corpus contracts
py -3.11 -m pytest -k merge
```

Two things about the `chunk` suite are worth knowing before the first failure:

* it runs from the **repository root**. Test modules import shared fixtures as
  top-level names (`from conftest import …`, `from _chunk_fixtures import …`),
  which is pytest's rootdir behaviour and is why `from tests.conftest import …`
  is not used — a `tests` package in an interpreter's site-packages shadows the
  repository's own and breaks collection.
* it uses deterministic tokenizer and embedding doubles, so it downloads no
  model. The real `cl100k_base` counter is covered separately in unit tests.

### The application suite

`tests/application` drives the product's use cases directly -- a container of
test doubles, an ordinary function call, an ordinary dict or an
`application.errors` exception back. No test client, no request context, no
application object anywhere in it, which is the point: it is the evidence that the
behaviour under the adapter is reusable, and `test_the_boundary_holds` fails
if any module under `application/` ever imports a web framework.

It does not repeat the API tests. Those prove an adapter still maps
correctly; this one proves there is something worth adapting -- which is what
made replacing the `/api/v1` adapter with FastAPI a change to one package.

```bash
python -m pytest tests/application -q
```

### The migration contract suite

A fourth thing to run, and the cheapest: `tests/migration` holds the
behaviours a platform migration must not change, written against what is
observable rather than against the implementation that satisfies it today.

```bash
python -m pytest tests/migration -q       # the suite
python -m pytest -m migration -q          # the same set, by marker
```

It is seconds, not minutes, on purpose -- it is meant to be run on every step
of a rewrite, not once at the end. Six files, one contract each:

| file | holds |
|---|---|
| `test_document_store_contract.py` | what a document store must do: the result record, the metadata round trip (including `search_text` / `table_view`), per-document isolation, pagination, durability -- and the twelve methods the routes call unguarded, which is more than `BaseVectorDB` declares. Run against **two** implementations: the shipped pgvector store and `reference_store.py`, a dependency-free store written to the contract and nothing else. Two is the point -- with one, a contract quietly becomes a description of that one. It was the runnable checklist pgvector was built against, and every test in it passed unchanged when the store underneath was replaced |
| `test_retrieval_parity.py` | the narrower claim a store migration needs: the shipped store and the reference implementation, over one corpus, answer with the **same ranking** -- top-k order, ties, filtering, identical-content chunks, empty results, deletion, re-indexing and the direction `distance` runs. Behavioural parity, not floating-point parity |
| `test_http_surface.py` | the console API is exactly what the README publishes, every route belongs to a declared group, and the refusal taxonomy (400/404/409/500/503/504) still makes all six distinctions -- read from the HTTP adapter's own translation tables |
| `test_domain_relations.py` | the edges between knowledge base, document, content and variant: what each deletion takes and what it must leave -- the foreign keys a schema has to declare |
| `test_api_v1_contract.py` | `/api/v1` as the contract it is: resource shapes, both identities, `visible = selected ∩ ready` on the wire, the refusal taxonomy, registry-driven method discovery, and that no module behind it keeps a method catalogue of its own |
| `test_legacy_removal_map.py` | `docs/legacy-removal.md` is the whole legacy surface and nothing else: every endpoint served is classified into exactly one removal wave, and every `/api/v1` replacement it names is a route that is really served. A plan that has gone stale fails here rather than in the step that trusted it |

What it deliberately leaves free: the web framework, the module layout, the
persistence, the vector store, the Viewer's implementation, and the internal
call graph. That freedom has been spent four times: `/api/v1` moved from
Flask to FastAPI, the record stores moved from JSON files to PostgreSQL, the
vector store moved from Chroma to pgvector, and the Flask surface beside it was
removed entirely -- each with **no change to any assertion in this directory**
about what the product does. The store migration edited two lines of
`test_document_store_contract.py`, both naming which implementations to run
against; every test body stayed as it was.

`tests/unit/test_fastapi_adapter.py` is the other half of that port -- the
questions the contract suite is blind to on purpose: the generated OpenAPI
document, the places FastAPI's defaults are bent to keep the contract (a bad
payload is 400 `invalid_request`, not 422; an unreadable page size is the
default, not a refusal), each refusal driven through the central table, and
the bridge that lets one process serve both surfaces.

### `/api/v1` over the real thing

The contract suite drives `/api/v1` against doubles, which is what makes it
blind to the framework and to the persistence -- and what stops it saying
whether any of it *works*. Three files in `tests/integration` say that, over
the tables the migrations built and the vectors pgvector stores, with only the
answer model and the embedding model replaced (`tests/api_v1_doubles.py`; they
are the two things that would otherwise leave the machine).

```bash
python -m pytest tests/integration/test_api_v1_end_to_end.py -q
python -m pytest tests/integration/test_api_v1_restart_persistence.py -q
python -m pytest tests/integration/test_api_v1_failure_and_concurrency.py -q
```

| file | holds |
|---|---|
| `test_api_v1_end_to_end.py` | the flows: a knowledge base created and renamed, an upload followed to its job and its stored chunks, an analysis built and a variant added, all three retrieval methods, a question answered with citations, an embedding index rebuilt in place, two knowledge bases that cannot see each other's corpus, and the two deletions |
| `test_api_v1_restart_persistence.py` | what a second process reads back: the records, the vectors, the manifest, the analysis and its rows, a job settled against the ledger, a staged upload swept -- and retrieval on a process that indexed nothing itself, because the lexical index is process-local and rebuilt rather than persisted |
| `test_api_v1_failure_and_concurrency.py` | the answers no seam can fake: a really full ingest queue refusing with `Retry-After`, the same bytes submitted twice at once becoming one parse, a delete racing an upload, a search running across a re-index, and a content two uploads share |

### The console suite

The front end has two suites, and the split is the same one this page draws
everywhere else: one that must pass with nothing running, and one that says
whether it *works*.

```bash
cd frontend
npm test               # everything, with no server anywhere
npm run typecheck      # tsc --noEmit
npm run build          # the production build
npm run test:live      # the real screens against a running console and server
```

`npm test` stubs `fetch` and holds the client to the contract's conventions --
the refusal taxonomy, the 204 with no body, `Retry-After`, walking a
collection -- and holds two architectural rules by reading the source: exactly
two modules call `fetch` (`lib/api/client.ts`, how a screen reaches the
contract, and `lib/api/proxy.ts`, how this server reaches the application), and
**no chunking method key is written down anywhere in the front end**. The
picker is rendered against a catalogue of invented method keys and every one of
them has to appear, which cannot pass if the component knows a real one.

`tests/proxy.test.ts` is the newest of these and holds one property: the
backend's address is resolved **per request**, not at build time. It was a
`rewrites()` entry in `next.config.mjs` until Step 14, and Next.js resolves
`rewrites()` during `next build` and freezes the destination into the build
output -- so the container image carried the developer default `127.0.0.1`,
which inside a container is that container, and `CHAT_RAG_API_URL` was read by
nothing. A green suite could not see it, and neither could a local run, because
the frozen value happens to be right on a developer's machine.

`npm run test:live` needs the application on its usual port and the console in
front of it (`npm run dev` or `npm start`, or the containers). It renders the shipped screens in
jsdom and drives the real flows: a knowledge base created from the dialog, a
document uploaded and its ingest job followed to `succeeded` by the shipped
poller, the analysis polled to `ready` and one method's rows inspected, a
search, a question answered with citations, and both deletions. It is not part
of `npm test` for the reason the sentence above gives.

### The guards worth knowing by name

These fail loudly and mean something specific:

| test | holds |
|---|---|
| `chunk/tests/unit/test_library_surface.py` | nothing on the library's product path imports research or legacy code — and names the import chain that broke it |
| `chat_rag/tests/unit/test_amsc_surface.py` | product code imports only `amsc.surface.CONSOLE_API` |
| `chat_rag/tests/unit/test_amsc_pin.py` | the pinned `amsc` revision provides every symbol the product imports, is on a remote branch, and the requirement line is shaped so pip can read it |
| `chat_rag/tests/unit/test_configuration.py` | the precedence rule, one owner per default, `env.example` cannot drift from the code, and every setting that is read is also applied |
| `chat_rag/tests/unit/test_provider_surface.py` | every shipped answer transport can be selected, is documented, and honours the query deadline |
| `chat_rag/tests/unit/test_state_isolation.py` | a test run cannot write to the developer's real state |
| `chunk/tests/unit/test_methods_registry.py` | the chunking-method registry is the one source of method identity, and a method registered in it reaches every consumer — including a Viewer page built before it existed |
| `chat_rag/tests/unit/test_chunker_extension.py` | one registration in the library is a console method: catalogue, API, upload, packager, Viewer routes — with no edit in this repository |
| `chat_rag/tests/unit/test_promote_chunk_pin.py` | the pin-promotion command moves only the sha, and refuses an unpushed, uncommitted or untracked revision |

---

## The smokes

Both are cheap enough to run in the image build, and both are run inside it.

```bash
python tools/import_smoke.py   # the declared dependencies satisfy every import
python tools/serve_smoke.py    # `python -m asgi` binds a socket, answers
                               # /api/v1/health on uvicorn, and stops on signal
```

A third tool sits beside them but answers a release question rather than a
build one: `python tools/promote_chunk_pin.py` moves the `amsc-poc` pin to a
`chunk` commit and checks it holds — see
[Changes that cross both repos](#changes-that-cross-both-repos).

`import_smoke` names the `amsc` that actually answered and, when pip recorded
one, the revision it came from — so "it works on my machine" becomes checkable.
Neither smoke downloads a model, contacts a provider or writes application
state: each runs against a throwaway data root.

## The Viewer shell build

The standalone Viewer page is a build artifact and is not in version control.
The product no longer uses it -- the Viewer is a screen of the console, at
`/viewer` -- so nothing in this repository builds it and `start-demo.ps1` does
not. Building it is only for serving the `chunk` repository's own frozen
corpus out of that checkout:

```bash
py -3.11 -m amsc.viewer.build --output artifacts/viewer-v3/index.html
```

With no `--benchmark` / `--deep` arguments this is the **shell**: no embedded
corpus, every document read live. It prints the path it wrote and how many
documents it embedded (`0` for the shell).

---

## The reproducibility gate

Does a clean clone work? Green tests do not answer that. They once stayed
green through a `requirements.txt` that could not be installed at all, because
every machine running them already had the package and an editable checkout of
the sibling library. So there is a separate gate, and it is one command:

```bash
python tools/verify_reproducibility.py
```

It clones this repository from the remote, checks the clone carries no state
from your machine, installs the pinned `amsc` revision into a fresh Python 3.11
environment, imports it, builds the Viewer v3 product shell from it, and then
brings up **the whole stack** from that clone with `docker compose up` --
database, application and console -- and asks it the two questions no health
check answers: was the schema built from an empty database by the container's
own migration step, and does `/api/v1` reach the application *through* the
console. Every check reports PASS, FAIL or SKIP -- SKIP means a capability is
missing (no Docker, no Python 3.11, no network) or a tier was not asked for,
never that something was checked and forgiven.

The gate runs *beside* your own stack, not through it: its own compose project,
its own published ports and its own image tags, so nothing it does touches the
containers or images this checkout built. It tears down with `-v`, because a
database volume left behind would let the next run's "built from empty" pass
for the wrong reason.

| Flag | What it adds |
|---|---|
| `--with-host-install` | installs `requirements.txt` and then `pip install --no-deps -e .` into a fresh venv on this machine as well (several minutes, ~1 GB of wheels) |
| `--local` | clones this checkout instead of the remote, to run the gate before pushing |
| `--no-docker` | skips the container checks |
| `--keep` | leaves the temporary clone and environments behind for inspection |

The same command runs in CI on every push to `main` and
`refactor/productionization` (`.github/workflows/reproducibility.yml`), so what
fails there fails here too, with the same output.

### Reading a gate result

```
PASS  clone                  origin/<branch> @ <sha>, the commit this checkout is on
PASS  pin.shape              amsc-poc @ <sha>, on a line of its own
PASS  chunk.install          pip installed <sha> into a fresh 3.11 venv
PASS  chunk.import           amsc <sha> from site-packages, not an editable sibling
PASS  chunk.viewer           shell built twice, byte-identical
PASS  docker.build           built from the clone; the import and serve smokes ran inside it
PASS  docker.run             compose up: three services healthy, schema built from empty
PASS  docker.console         the console forwards /api/v1 to the application by service name
PASS  docker.state           state under /data only, none in /app
PASS  state.untouched        the developer's data is unchanged
```

`chunk.import` saying "not an editable sibling" is the whole point: on a
developer machine `amsc` is an editable install of `../chunk`, so every import
succeeds whatever the pin says. The gate refuses that shortcut.

What each failure means is in
[operations.md](operations.md).

---

## Changes that cross both repos

`chat_rag` installs `amsc` from an **immutable commit**. A library change is
therefore not done when the library's tests pass — it is done when the console
is pinned to it. The order matters, and skipping a step fails somewhere later
and less clearly:

```
1.  edit chunk/                     add the method, fix the bug
2.  py -3.11 -m pytest              the chunk suite, green
3.  git commit && git push          the pin must name a commit that exists
                                    on a remote, or no clean install can fetch it
4.  python tools/promote_chunk_pin.py          steps 4 and 5, as one command:
                                    it resolves ../chunk HEAD, refuses a
                                    revision that is unpushed or that HEAD does
                                    not contain (an untracked new file included),
                                    rewrites the sha in requirements.txt and
                                    runs tests/unit/test_amsc_pin.py, putting
                                    the file back if that fails
5.  python -m pytest -q             the rest of the chat_rag suite
6.  python tools/verify_reproducibility.py     the declared source installs
```

`promote_chunk_pin.py` takes `--rev` (a revision other than `HEAD`), `--check`
(report and write nothing), `--push` (push the chunk checkout's branch first —
its only network call) and `--chunk-repo`. It never commits this repository;
it prints the `git` command for what it changed.

Notes that save an afternoon:

* **Locally, `amsc` is an editable install of `../chunk`.** Your working tree
  is what the console imports, so a chunk change appears to work before it is
  committed, pushed or pinned. Only the gate and a clean install disagree.
* **The pin line must end at the sha.** A lost newline once merged the next
  requirement into the git URL, which made `pip install -r requirements.txt`
  fail outright while every suite stayed green. `test_amsc_pin.py` checks the
  line's shape for exactly this reason, and `promote_chunk_pin.py` rewrites
  only the sha of that one line so the shape cannot be lost by hand again.
* **Adding a name to `amsc.surface.CONSOLE_API` is part of step 1**, not an
  afterthought: the console may import nothing else, and the guard on both
  sides reads the declaration from the installed library.
* If the change touches the Viewer **template**, the page is rebuilt from the
  new library — the page *is* the template. Adding a chunking method is not
  such a change: a served page reads the method registry from its own server
  (`GET /api/methods`) at boot, so a new method appears without a rebuild. See
  [chunk/docs/viewer-architecture.md](../../chunk/docs/viewer-architecture.md).

---

## Before calling a change done

The list this project actually runs:

```bash
# chunk
py -3.11 -m pytest

# chat_rag
python -m pytest -q
python tools/import_smoke.py
python tools/serve_smoke.py
python tools/verify_reproducibility.py        # or --local before pushing
```

and, when the change touched the Viewer or the library boundary:

```bash
py -3.11 -m amsc.viewer.build --output artifacts/viewer-v3/index.html   # in chunk
py -3.11 -m pytest tests/unit/test_library_surface.py                # in chunk
python -m pytest -q tests/unit/test_amsc_surface.py tests/unit/test_amsc_pin.py
```

The suites print a warning line if a run changed the developer's knowledge
bases, ledger or gold set. That line should never appear; if it does, treat it
as a bug in the test isolation rather than as noise.
