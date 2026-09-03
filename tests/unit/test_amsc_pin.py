"""Does the ``amsc`` revision pinned in requirements.txt provide what the
product imports?

Locally ``amsc`` is an editable install of the sibling ``chunk`` checkout, so
``import app`` succeeds whatever the pin says; a clean install (the
Dockerfile) gets exactly the pinned commit. This test asks the pinned commit
itself, through the sibling checkout's git history, whether every
``from amsc.<module> import <name>`` in the product code resolves there. It
needs no network and no fresh environment.

It is marked ``xfail(strict=True)``: the pin is known to be stale, and the
marker documents that. Bumping the pin to a revision that carries every
symbol makes the test pass, at which point ``strict`` demands the marker go.
Without a ``chunk`` checkout beside this one (or ``CHUNK_REPO``) the test
skips rather than guessing.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "requirements.txt"
PIN = re.compile(r"amsc-poc\s*@\s*git\+\S+?@([0-9a-f]{7,40})")
IMPORT = re.compile(r"^\s*from\s+(amsc(?:\.[\w]+)*)\s+import\s+([^\n#]+)", re.M)
STALE_PIN = "Phase 1B: requirements.txt pins amsc 8222a21, which predates table_view, table_search_text, deep_arm.package_arm and viewer_v3"


def _chunk_repo() -> Path | None:
    candidates = [os.environ.get("CHUNK_REPO"), str(ROOT.parent / "chunk")]
    for candidate in candidates:
        if candidate and (Path(candidate) / ".git").exists():
            return Path(candidate)
    return None


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")


def _product_imports() -> dict[str, set[str]]:
    wanted: dict[str, set[str]] = {}
    for path in ROOT.rglob("*.py"):
        parts = set(path.relative_to(ROOT).parts)
        if parts & {"tests", "venv", ".venv", "examples", "__pycache__"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for module, names in IMPORT.findall(text):
            for name in names.replace("(", "").replace(")", "").split(","):
                name = name.strip().split(" as ")[0].strip()
                if name:
                    wanted.setdefault(module, set()).add(name)
    return wanted


def _pinned_commit() -> str:
    match = PIN.search(REQUIREMENTS.read_text(encoding="utf-8"))
    assert match, "requirements.txt no longer pins amsc-poc to a git commit"
    return match.group(1)


def _missing_at(repo: Path, commit: str, wanted: dict[str, set[str]]) -> list[str]:
    missing = []
    for module, names in sorted(wanted.items()):
        path = "src/" + module.replace(".", "/") + ".py"
        shown = _git(repo, "show", f"{commit}:{path}")
        if shown.returncode != 0:
            package = _git(repo, "show", f"{commit}:src/{module.replace('.', '/')}/__init__.py")
            if package.returncode != 0:
                missing.append(f"{module} (module absent)")
                continue
            shown = package
        source = shown.stdout
        for name in sorted(names):
            defined = re.search(rf"^(?:def|class)\s+{re.escape(name)}\b|^{re.escape(name)}\s*[:=]", source, re.M)
            # ``from amsc import hybrid_chunker`` names a submodule, not a symbol.
            submodule = _git(repo, "cat-file", "-e", f"{commit}:src/{module.replace('.', '/')}/{name}.py")
            if not defined and submodule.returncode != 0:
                missing.append(f"{module}.{name}")
    return missing


@pytest.fixture
def chunk_repo():
    repo = _chunk_repo()
    if repo is None:
        pytest.skip("no chunk checkout beside this one (set CHUNK_REPO)")
    if _git(repo, "rev-parse", "--verify", "--quiet", f"{_pinned_commit()}^{{commit}}").returncode != 0:
        pytest.skip("the pinned commit is not in the local chunk checkout")
    return repo


def test_the_product_imports_a_known_set_of_amsc_symbols():
    """The list the pin is checked against; a new import shows up here."""
    wanted = _product_imports()
    assert "amsc.table_view" in wanted and "CONTEXT_HEADER" in wanted["amsc.table_view"]
    assert "amsc.deep_arm" in wanted and {"package", "package_arm"} <= wanted["amsc.deep_arm"]
    assert "amsc.deep_pipeline" in wanted and "chunk_document" in wanted["amsc.deep_pipeline"]


@pytest.mark.xfail(strict=True, reason=STALE_PIN)
def test_the_pinned_amsc_revision_provides_every_symbol_the_product_imports(chunk_repo):
    missing = _missing_at(chunk_repo, _pinned_commit(), _product_imports())
    assert missing == [], f"pinned {_pinned_commit()[:7]} lacks: {missing}"


def test_the_checked_out_amsc_revision_provides_every_symbol_the_product_imports(chunk_repo):
    """The same check against HEAD of the sibling checkout, which is what the
    editable install actually serves -- this is the masking, made visible."""
    head = _git(chunk_repo, "rev-parse", "HEAD").stdout.strip()
    assert _missing_at(chunk_repo, head, _product_imports()) == []
