"""What this console is allowed to import from the chunking library.

``amsc`` is one flat namespace holding product code, research experiments and
legacy compatibility side by side. Which is which is declared in
:mod:`amsc.surface` and enforced there against the library's own import graph;
this is the other half of that contract -- the console's side.

Two rules, deliberately different in strength:

* **product code** may import only :data:`amsc.surface.CONSOLE_API`. That set
  is the console contract: twenty-one modules the console genuinely calls. An
  import outside it is either a new dependency that should be declared, or a
  reach into a library internal that will break on the next pin bump.
* **tests** may import any *product* module -- fixtures legitimately need
  things the console itself does not, such as ``amsc.units`` for building a
  frozen embedding snapshot -- but never a research, legacy or unused one.

Why not one rule for both: tightening the test side to ``CONSOLE_API`` would
either bloat that set with fixture-only modules, which makes the contract less
useful, or push fixtures into contortions. What actually matters is that no
test quietly normalises a research import that product code then copies.

This test reads the declaration from the installed ``amsc``, so a pin bump
that changes the library's own boundary is felt here immediately rather than
at the next deploy.
"""

from __future__ import annotations

import ast
import pkgutil
from pathlib import Path

import pytest

import amsc
from amsc import surface

REPO = Path(__file__).resolve().parents[2]

#: Every submodule the installed library actually has, so ``from amsc import X``
#: can be told apart from ``from amsc import SomeClass`` -- the package exports
#: both, and only the first is an import of a module.
SUBMODULES = frozenset(info.name for info in pkgutil.iter_modules(amsc.__path__))

#: Everything that ships and runs. Anything not here is a test or a fixture.
PRODUCT_TREES = ("app.py", "wsgi.py", "main_new.py", "setup_nltk.py",
                 "components", "config", "core", "cli", "pipeline", "utils",
                 "evaluation", "tools", "examples")

#: Directories no scan should walk into.
SKIP = {"venv", ".venv", "__pycache__", "artifacts", "chroma_db", "faiss_db",
        ".cache", ".demo", ".docker-data", ".git", "node_modules"}


def _python_files(*relatives: str):
    for relative in relatives:
        path = REPO / relative
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*.py")):
                if not any(part in SKIP for part in child.parts):
                    yield child


def _amsc_imports(path: Path) -> set[str]:
    """The ``amsc`` submodules one file imports, from its AST.

    The bare package (``import amsc``) is not a submodule import and is always
    allowed: it is how the provenance snapshot reads the library's version.
    """
    found: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - a file that does not parse
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "amsc":
                # ``from amsc import methods`` names a submodule;
                # ``from amsc import V4Chunker`` names a class the package
                # re-exports. Only the first is an import of a module.
                found |= {a.name for a in node.names if a.name in SUBMODULES}
            elif node.module.startswith("amsc."):
                found.add(node.module.split(".", 1)[1].split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("amsc."):
                    found.add(alias.name.split(".", 1)[1].split(".")[0])
    return found


def _scan(*relatives: str) -> dict[str, set[str]]:
    """``{module: {files that import it}}`` over the given trees."""
    out: dict[str, set[str]] = {}
    for path in _python_files(*relatives):
        for module in _amsc_imports(path):
            out.setdefault(module, set()).add(path.relative_to(REPO).as_posix())
    return out


PRODUCT_IMPORTS = _scan(*PRODUCT_TREES)
TEST_IMPORTS = _scan("tests")


# ------------------------------------------------------------ product code


def test_product_code_imports_only_the_declared_console_api():
    """The console contract, checked against every shipping file.

    A failure here is a decision, not a chore: either the import belongs in
    ``amsc.surface.CONSOLE_API`` -- add it there, in the chunk repository, and
    bump the pin -- or the console should be reaching for something else.
    """
    outside = {
        module: sorted(files)
        for module, files in sorted(PRODUCT_IMPORTS.items())
        if module not in surface.CONSOLE_API
    }
    assert outside == {}, "\n".join(
        ["product code imports amsc modules outside the console API:"]
        + [f"  amsc.{module}  ({surface.classify(module)})  <- {', '.join(files)}"
           for module, files in outside.items()]
    )


def test_product_code_reaches_no_research_or_legacy_module():
    """The same rule stated as the thing it is protecting against.

    Kept separate from the test above so a failure says *which* kind of
    mistake was made: a new but legitimate dependency, or a reach into an
    experiment.
    """
    forbidden = {
        module: sorted(files)
        for module, files in sorted(PRODUCT_IMPORTS.items())
        if surface.classify(module) in ("research", "legacy", "unused")
    }
    assert forbidden == {}, "\n".join(
        ["product code imports research or legacy library modules:"]
        + [f"  amsc.{module}  ({surface.classify(module)})  <- {', '.join(files)}"
           for module, files in forbidden.items()]
    )


def test_the_console_api_carries_nothing_the_console_does_not_use():
    """The contract stays honest in both directions.

    A name left in ``CONSOLE_API`` after its last caller goes reads as a
    supported dependency that nobody is checking. ``deep_arm`` and
    ``viewer_corpus`` are in the set because the Viewer packaging calls them,
    ``structural_qa`` because the report CLI does, and so on -- each entry
    should be traceable to a file.
    """
    unused = sorted(surface.CONSOLE_API - set(PRODUCT_IMPORTS))
    assert unused == [], (
        "these are declared as console API but no product file imports them; "
        f"remove them from amsc/surface.py or use them: {unused}"
    )


# ------------------------------------------------------------------- tests


def test_tests_import_no_research_or_legacy_module():
    """Fixtures get the whole product surface, and nothing beyond it."""
    forbidden = {
        module: sorted(files)
        for module, files in sorted(TEST_IMPORTS.items())
        if surface.classify(module) in ("research", "legacy", "unused")
    }
    assert forbidden == {}, "\n".join(
        ["tests import research or legacy library modules:"]
        + [f"  amsc.{module}  ({surface.classify(module)})  <- {', '.join(files)}"
           for module, files in forbidden.items()]
    )


def test_the_scan_actually_found_the_imports():
    """A guard that finds nothing guards nothing."""
    assert len(PRODUCT_IMPORTS) >= 15, PRODUCT_IMPORTS
    assert "deep_pipeline" in PRODUCT_IMPORTS
    assert "viewer_corpus" in PRODUCT_IMPORTS
    assert PRODUCT_IMPORTS["viewer_corpus"] == {"components/viewer/analysis.py"}
    assert "methods" in PRODUCT_IMPORTS, "the registry import must be seen"


@pytest.mark.parametrize("module", sorted(surface.CONSOLE_API))
def test_every_console_api_module_imports(module):
    """The declared contract resolves against the installed library."""
    __import__(f"amsc.{module}")
