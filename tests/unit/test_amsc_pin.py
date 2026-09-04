"""Does the ``amsc`` revision pinned in requirements.txt provide what the
product imports?

Locally ``amsc`` is an editable install of the sibling ``chunk`` checkout, so
``import app`` succeeds whatever the pin says; a clean install (the
Dockerfile) gets exactly the pinned commit. This test asks the pinned commit
itself, through the sibling checkout's git history, whether every
``from amsc.<module> import <name>`` in the product code resolves there. It
needs no network and no fresh environment.

Without a ``chunk`` checkout beside this one (or ``CHUNK_REPO``) the tests
skip rather than guessing.

A pin can be wrong in three ways, and all three are checked: it can name a
revision that lacks a symbol the product imports (the failure Phase 1A
reproduced); it can name a revision that exists only on the developer's machine
-- which breaks a clean install exactly as visibly, and only when someone else
builds; or the line itself can be malformed, which is what happened next.
Phase 1B's own pin edit lost the newline after the commit, so the file read

    amsc-poc @ git+https://github.com/erenayd58/chunk.git@<sha>openai

-- a git ref that does not exist, and one dependency (``openai``) swallowed
into it. ``pip install -r requirements.txt`` failed outright, so no clean
install and no image build was possible at all; a checkout with an editable
``amsc`` and ``openai`` already installed noticed nothing, and the pin test
above passed because its regex was happy to stop after 40 hex characters.
So the file is now also checked for being *installable-shaped*, not just for
naming a good commit.
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


def test_the_pinned_amsc_revision_provides_every_symbol_the_product_imports(chunk_repo):
    missing = _missing_at(chunk_repo, _pinned_commit(), _product_imports())
    assert missing == [], f"pinned {_pinned_commit()[:7]} lacks: {missing}"


def test_the_checked_out_amsc_revision_provides_every_symbol_the_product_imports(chunk_repo):
    """The same check against HEAD of the sibling checkout, which is what the
    editable install actually serves -- this is the masking, made visible."""
    head = _git(chunk_repo, "rev-parse", "HEAD").stdout.strip()
    assert _missing_at(chunk_repo, head, _product_imports()) == []


def test_the_pinned_revision_is_one_a_clean_install_can_actually_fetch(chunk_repo):
    """A pin only has to exist *somewhere* to satisfy the symbol check above,
    and a commit that was never pushed satisfies it on this machine and on no
    other. Checked against the local remote-tracking refs, so it needs no
    network; it skips when the checkout has none."""
    commit = _pinned_commit()
    remotes = _git(chunk_repo, "branch", "-r", "--contains", commit)
    if remotes.returncode != 0:
        pytest.skip("this chunk checkout has no remote-tracking refs to check against")
    branches = [line.strip() for line in remotes.stdout.splitlines() if line.strip()]
    assert branches, (
        f"pinned {commit[:7]} is on no remote branch in this checkout: a clean "
        "install cannot fetch it. Push the branch, or pin a revision that is pushed."
    )


# ------------------------------------------------- the line, as pip reads it

PIN_LINE = re.compile(r"^amsc-poc\s*@\s*(\S+)\s*$", re.M)


def test_the_pin_line_ends_at_the_commit():
    """A requirement is one line. Losing the newline joins it to the next one.

    ``git+<url>@<sha>openai`` is a syntactically valid requirement naming a
    revision that will never resolve, and it silently costs whichever
    requirement followed it.
    """
    text = REQUIREMENTS.read_text(encoding="utf-8")
    match = PIN_LINE.search(text)
    assert match, "requirements.txt no longer declares amsc-poc on a line of its own"

    url = match.group(1)
    revision = url.rsplit("@", 1)[-1]
    assert re.fullmatch(r"[0-9a-f]{40}", revision), (
        f"the amsc pin ends in {revision!r}, which is not a bare commit sha -- "
        "the line has run into whatever follows it"
    )


def test_every_requirement_is_a_line_pip_can_read():
    """No stray joins anywhere else in the file, either."""
    problems = []
    for number, line in enumerate(REQUIREMENTS.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = re.split(r"[\s<>=!@\[;#]", stripped, maxsplit=1)[0]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            problems.append(f"line {number}: {stripped!r}")
    assert not problems, "requirements.txt lines pip cannot parse as a name: " + "; ".join(problems)


def test_the_dependencies_the_product_imports_are_declared():
    """``openai`` was a declared dependency until a newline went missing."""
    declared = {
        re.split(r"[\s<>=!@\[;#]", line.strip(), maxsplit=1)[0].lower()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    # Each of these is imported unconditionally by a module the application
    # loads: the Azure client, the web framework, and the production server.
    for required in ("openai", "flask", "waitress"):
        assert required in declared, f"{required} is imported but no longer declared"

