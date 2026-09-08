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
python -m pytest tests/unit -q            # fast: no Flask app, no store
python -m pytest tests/application -q     # the product's behaviour, with no Flask at all
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
Flask object anywhere in it, which is the point: it is the evidence that the
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
of a rewrite, not once at the end. Four files, one contract each:

| file | holds |
|---|---|
| `test_document_store_contract.py` | what a document store must do: the result record, the metadata round trip (including `search_text` / `table_view`), per-document isolation, pagination, durability -- and the twelve methods the routes call unguarded, which is more than `BaseVectorDB` declares. Run against **two** implementations: the shipped Chroma store and `reference_store.py`, a dependency-free store written to the contract and nothing else. Two is the point -- with one, a contract quietly becomes a description of that one, and `reference_store.py` doubles as the runnable checklist for pgvector |
| `test_http_surface.py` | the console API is exactly what the README publishes, every route belongs to a declared group, and the refusal taxonomy (400/404/409/500/503/504) still makes all six distinctions -- read from the HTTP adapter's own translation tables |
| `test_domain_relations.py` | the edges between knowledge base, document, content and variant: what each deletion takes and what it must leave -- the foreign keys a schema has to declare |
| `test_api_v1_contract.py` | `/api/v1` as the contract it is: resource shapes, both identities, `visible = selected ∩ ready` on the wire, the refusal taxonomy, registry-driven method discovery, and that no module behind it keeps a method catalogue of its own |

What it deliberately leaves free: the web framework, the module layout, the
file-backed persistence, Chroma, the Viewer's implementation, and the internal
call graph. That freedom has been spent once already: `/api/v1` moved from
Flask to FastAPI with **no change to any file in this directory**, which is
the strongest thing that can be said about a contract suite.

`tests/unit/test_fastapi_adapter.py` is the other half of that port -- the
questions the contract suite is blind to on purpose: the generated OpenAPI
document, the places FastAPI's defaults are bent to keep the contract (a bad
payload is 400 `invalid_request`, not 422; an unreadable page size is the
default, not a refusal), each refusal driven through the central table, and
the bridge that lets one process serve both surfaces.

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
python tools/serve_smoke.py    # `python -m wsgi` binds a socket, answers
                               # /api/health on waitress, and stops on signal
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

The Viewer page is a build artifact and is not in version control. From the
`chunk` checkout:

```bash
py -3.11 -m amsc.viewer.build --output artifacts/viewer-v3/index.html
```

With no `--benchmark` / `--deep` arguments this builds the **product shell**:
no embedded corpus, every document read live from the console. That is the
build a fresh clone can always make, and the one `start-demo.ps1` runs when
the page is missing. It prints the path it wrote and how many documents it
embedded (`0` for the shell).

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
environment, imports it, builds the Viewer v3 product shell from it, then
builds and runs the container and asks it for `/api/health`. Every check
reports PASS, FAIL or SKIP -- SKIP means a capability is missing (no Docker, no
Python 3.11, no network) or a tier was not asked for, never that something was
checked and forgiven.

| Flag | What it adds |
|---|---|
| `--with-host-install` | installs `requirements.txt` into a fresh venv on this machine as well (several minutes, ~1 GB of wheels) |
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
PASS  docker.build / docker.run / docker.state
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
bases, ledger, gold set or Chroma store. That line should never appear; if it
does, treat it as a bug in the test isolation rather than as noise.
