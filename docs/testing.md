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

```bash
# chat_rag (this repo) — the venv's interpreter, from the repo root
python -m pytest -q                       # everything
python -m pytest tests/unit -q            # fast: no Flask app, no store
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
| `chunk/tests/unit/test_methods_registry.py` | the chunking-method registry is the one source of method identity |

---

## The smokes

Both are cheap enough to run in the image build, and both are run inside it.

```bash
python tools/import_smoke.py   # the declared dependencies satisfy every import
python tools/serve_smoke.py    # `python -m wsgi` binds a socket, answers
                               # /api/health on waitress, and stops on signal
```

`import_smoke` names the `amsc` that actually answered and, when pip recorded
one, the revision it came from — so "it works on my machine" becomes checkable.
Neither smoke downloads a model, contacts a provider or writes application
state: each runs against a throwaway data root.

## The Viewer shell build

The Viewer page is a build artifact and is not in version control. From the
`chunk` checkout:

```bash
py -3.11 -m amsc.viewer_v3 --output artifacts/viewer-v3/index.html
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
4.  edit chat_rag/requirements.txt  amsc-poc @ git+…@<the new 40-char sha>
5.  python -m pytest -q             the chat_rag suite, green — test_amsc_pin.py
                                    now checks the new revision provides every
                                    symbol the product imports
6.  python tools/verify_reproducibility.py     the declared source installs
```

Notes that save an afternoon:

* **Locally, `amsc` is an editable install of `../chunk`.** Your working tree
  is what the console imports, so a chunk change appears to work before it is
  committed, pushed or pinned. Only the gate and a clean install disagree.
* **The pin line must end at the sha.** A lost newline once merged the next
  requirement into the git URL, which made `pip install -r requirements.txt`
  fail outright while every suite stayed green. `test_amsc_pin.py` checks the
  line's shape for exactly this reason.
* **Adding a name to `amsc.surface.CONSOLE_API` is part of step 1**, not an
  afterthought: the console may import nothing else, and the guard on both
  sides reads the declaration from the installed library.
* If the change touches the Viewer, the release sequence has one more step —
  the page is rebuilt from the new library. See
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
py -3.11 -m amsc.viewer_v3 --output artifacts/viewer-v3/index.html   # in chunk
py -3.11 -m pytest tests/unit/test_library_surface.py                # in chunk
python -m pytest -q tests/unit/test_amsc_surface.py tests/unit/test_amsc_pin.py
```

The suites print a warning line if a run changed the developer's knowledge
bases, ledger, gold set or Chroma store. That line should never appear; if it
does, treat it as a bug in the test isolation rather than as noise.
