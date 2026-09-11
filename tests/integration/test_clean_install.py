"""The distribution, installed where nothing else is, and used for the first time.

    python -m build  ->  pip install <wheel | sdist> into an empty venv
                     ->  import, migrate, ingest, search  (in that interpreter)

``tests/unit/test_distribution.py`` reads the built wheel and says what is in
it. This is the claim after that one, and the one a consumer actually
depends on: that the artifact **installs** with nothing named but itself --
``amsc-poc`` is on no index, and the wheel's own metadata has to say where it
comes from -- and that what installed **works**: builds its own schema in an
empty PostgreSQL through the public API, ingests a Markdown document with no
extra and no provider configured, and searches it.

Both artifacts, because they are installed differently. A wheel is unpacked;
an sdist is *built* first, in an isolated environment that fetches the build
backend, and a file the sdist forgot -- a migration, a YAML, ``py.typed`` --
is a wheel that builds and a library that does not run.

The mechanics are ``tools/wheel_smoke.py``'s, driven from here rather than
copied, so the gate (``tools/verify_reproducibility.py``) and the suite ask
the same questions of the same functions. Each artifact gets its own empty
database (``fresh_database``) and its own throwaway interpreter, removed
afterwards rather than left in pytest's kept temporary directories.

What it needs: a network, for the pinned ``amsc-poc`` commit and the index;
``git``, which pip uses to fetch it; and a few minutes. Deselect it with
``-k "not clean_install"`` when offline.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from tools import wheel_smoke as smoke


@pytest.fixture(scope="module")
def artifacts() -> dict[str, Path]:
    """One wheel and one sdist, built from this checkout the way a release
    would be, once for the module."""
    out = Path(tempfile.mkdtemp(prefix="chat_rag-dist-"))
    report = smoke.Report()
    built = smoke.build(report, out, sdist=True)
    assert report.failures == [] and set(built) == {"wheel", "sdist"}, (
        f"the distribution did not build: {report.failures}")
    yield built
    shutil.rmtree(out, ignore_errors=True)


@pytest.fixture
def clean_interpreter():
    """Somewhere to create a venv that is gone when the test is."""
    where = Path(tempfile.mkdtemp(prefix="chat_rag-clean-"))
    yield where
    shutil.rmtree(where, ignore_errors=True)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_the_artifact_installs_alone_and_its_first_use_works(
        kind, artifacts, clean_interpreter, fresh_database):
    """The whole consumer experience, in the consumer's own interpreter."""
    report = smoke.Report()

    # Installs naming nothing but the artifact: the direct reference in the
    # metadata is what fetches amsc-poc, and this is where that is tested.
    python = smoke.install(report, f"{kind}.install", clean_interpreter / "venv",
                           str(artifacts[kind]))
    assert python is not None, f"{kind} did not install: {report.failures}"

    found = smoke.probe(report, python, expect_heavy=False,
                        database_url=fresh_database, prefix=kind)
    assert found is not None, f"the installed {kind} could not be used: {report.failures}"
    assert report.failures == [], f"the installed {kind} fell short: {report.failures}"

    # The claims the report already checked, restated as the assertions a
    # reader looks for: the surface, the schema, the first use.
    assert found["published"] == found["api_published"] == found["resolves"]
    assert found["heavy_after_import"] == [] and found["heavy_after_first_use"] == []
    assert found["migration"] == ["created", "current"]
    assert found["first_use"]["knowledge_bases"] == ["Smoke"]
    assert found["first_use"]["chunks"] > 0 and found["first_use"]["hits"] > 0
    assert found["first_use"]["top_hit_mentions_liquidity"] is True
    assert found.get("leaked") is None
