"""What is in the wheel, which is where the library boundary is actually kept.

Every other statement of the boundary is a convention: a docstring saying the
adapter lives outside the package, a test asserting no use case imports a web
framework. This one is the artifact. ``pip install chat-rag`` gets exactly what
is checked here, so a consumer cannot reach ``interfaces.http`` by accident and
cannot fail to reach ``chat_rag.storage.migrations`` when they need a schema.

Two failures this file exists for, and both were real:

* **the migrations did not ship.** ``migrations/`` and ``migrations/versions/``
  have no ``__init__.py`` -- Alembic loads ``env.py`` and each revision by path
  -- so ``packages.find`` never saw them, and a wheel built before L5 installed
  a library that could not create its own schema. They are package data now,
  and this says so from the outside rather than trusting the declaration;
* **the adapter could ship.** ``interfaces/``, ``asgi.py``, ``cli/`` and
  ``tools/`` are outside the package for a reason -- so that FastAPI cannot
  become a dependency of the engine -- and one ``packages.find`` entry pointing
  at the repository root would quietly undo it.

The wheel is built once for the module. It costs a few seconds and it is the
only honest way to ask the question: reading ``pyproject.toml`` back would
check that the declaration says what it says.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Every module a consumer must be able to import, spelled as a path inside
#: the wheel. Not an exhaustive list of the package -- that would be a copy of
#: the source tree -- but one file from each thing the library is made of, so
#: a package dropped from ``packages.find`` fails here.
MUST_SHIP = (
    "chat_rag/__init__.py",
    "chat_rag/api/__init__.py",
    "chat_rag/api/engine.py",
    "chat_rag/api/results.py",
    "chat_rag/application/services.py",
    "chat_rag/components/chunker/factory.py",
    "chat_rag/config/settings.py",
    "chat_rag/core/exceptions.py",
    "chat_rag/pipeline/rag_pipeline.py",
    "chat_rag/runtime.py",
    "chat_rag/storage/models.py",
    "chat_rag/utils/document_tracker.py",
    # Data, not modules, and each one is read by path at run time.
    "chat_rag/py.typed",
    "chat_rag/config/frozen_v4.yaml",
    "chat_rag/config/benchmark_aligned_retrieval.yaml",
    # The schema. A library that cannot create its own tables is not
    # installable in any useful sense.
    "chat_rag/storage/migrations/env.py",
    "chat_rag/storage/migrations/script.py.mako",
    "chat_rag/storage/migrations/versions/0001_initial_schema.py",
    "chat_rag/storage/migrations/versions/0002_pgvector_store.py",
)

#: Top-level names that must not appear in the wheel. The adapter, its entry
#: point, the operator programs, the console and the tests: each of them
#: imports this package, and none of them is it.
MUST_NOT_SHIP = (
    "interfaces", "asgi", "cli", "tools", "runtime/bootstrap", "frontend",
    "tests", "evaluation", "alembic.ini", "docker-compose", "Dockerfile",
)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> zipfile.ZipFile:
    """One wheel, built from this checkout the way a release would be.

    ``--no-isolation`` because the build backend is already installed here and
    a fresh isolated environment would download setuptools to prove a claim
    about *this* repository's packaging. The clean-environment question --
    does the built wheel install and import with nothing else around -- is a
    different one, and ``tools/wheel_smoke.py`` answers it.
    """
    out = tmp_path_factory.mktemp("wheel")
    built = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(out), str(REPO)],
        capture_output=True, text=True, timeout=900,
    )
    if built.returncode != 0:
        pytest.fail("building the wheel failed:\n" + built.stdout[-3000:]
                    + "\n" + built.stderr[-3000:])
    files = sorted(out.glob("*.whl"))
    assert len(files) == 1, f"expected one wheel, got {[f.name for f in files]}"
    with zipfile.ZipFile(files[0]) as archive:
        archive.built_name = files[0].name
        yield archive


@pytest.fixture(scope="module")
def contents(wheel) -> list[str]:
    return wheel.namelist()


# ------------------------------------------------------------- what is in it
@pytest.mark.parametrize("member", MUST_SHIP)
def test_the_library_ships_what_a_consumer_has_to_import(contents, member):
    assert member in contents, f"{member} is not in the wheel"


def test_the_wheel_is_the_engine_and_only_the_engine(contents):
    """Every module in it belongs to ``chat_rag``; the rest is metadata."""
    strays = [name for name in contents
              if not name.startswith("chat_rag/")
              and not name.startswith("chat_rag-")  # .dist-info
              and not name.endswith("/")]
    assert strays == [], f"the wheel carries files outside the package: {strays}"


@pytest.mark.parametrize("excluded", MUST_NOT_SHIP)
def test_what_runs_beside_the_library_is_not_in_it(contents, excluded):
    """The boundary, as the artifact rather than as a convention.

    ``interfaces/`` in the wheel would make FastAPI reachable from a library
    install, which is exactly what putting it outside the package prevents.
    """
    present = [name for name in contents
               if name == excluded or name.startswith(excluded + "/")
               or name.startswith(excluded + ".")]
    assert present == [], f"{excluded} is in the wheel: {present}"


def test_no_web_framework_is_declared_as_a_dependency(wheel):
    """The other half of the same claim, read off the metadata.

    A wheel that shipped only ``chat_rag`` but declared ``fastapi`` would put
    the framework in every consumer's environment without ever importing it.
    """
    metadata = _metadata(wheel)
    requires = [line.split(":", 1)[1].strip().lower()
                for line in metadata.splitlines() if line.startswith("Requires-Dist:")]
    for framework in ("fastapi", "uvicorn", "starlette", "flask", "python-multipart"):
        offending = [line for line in requires if line.split()[0].split("[")[0] == framework]
        assert offending == [], f"the wheel requires {framework}: {offending}"


# ---------------------------------------------------------------- the extras
def test_postgresql_is_required_and_not_an_extra(wheel):
    """There is no mode of this engine that runs without a database, so the
    four packages that reach one are core dependencies -- not something a
    consumer can leave out and discover at the first query."""
    core = _core_requirements(wheel)
    for required in ("sqlalchemy", "psycopg", "alembic", "pgvector"):
        assert any(line.startswith(required) for line in core), (
            f"{required} is not a core dependency: {core}")


def test_the_heavy_dependencies_are_extras(wheel):
    """What a consumer is allowed not to download.

    torch (through sentence-transformers) is most of a gigabyte and the PDF
    stack is a large native dependency; a deployment that embeds through a
    gateway and ingests Markdown needs neither.
    """
    core = _core_requirements(wheel)
    for optional in ("sentence-transformers", "ollama", "pymupdf",
                     "pymupdf4llm", "python-docx"):
        assert not any(line.startswith(optional) for line in core), (
            f"{optional} is still a core dependency")

    metadata = _metadata(wheel)
    assert "Provides-Extra: local" in metadata
    assert "Provides-Extra: pdf" in metadata
    assert "Provides-Extra: all" in metadata


def test_every_extra_is_reachable_through_all(wheel):
    """``pip install chat-rag[all]`` has to mean every extra, or it is a third
    thing to keep in step by hand."""
    metadata = _metadata(wheel)
    extras = {line.split(":", 1)[1].strip() for line in metadata.splitlines()
              if line.startswith("Provides-Extra:")}
    umbrella = _requirements_for(wheel, "all")
    for extra in extras - {"all"}:
        assert any(f"[{extra}]" in line or f'extra == "{extra}"' in line
                   for line in umbrella), f"[all] does not include [{extra}]"


def _metadata(wheel) -> str:
    name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
    return wheel.read(name).decode("utf-8")


def _requirements(wheel) -> list[str]:
    return [line.split(":", 1)[1].strip()
            for line in _metadata(wheel).splitlines()
            if line.startswith("Requires-Dist:")]


def _core_requirements(wheel) -> list[str]:
    """The dependencies installed with no extras asked for."""
    return [line.lower() for line in _requirements(wheel) if "extra ==" not in line]


def _requirements_for(wheel, extra: str) -> list[str]:
    return [line for line in _requirements(wheel) if f'extra == "{extra}"' in line]
