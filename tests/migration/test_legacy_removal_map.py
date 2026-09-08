"""The removal actually happened, and cannot come back by accident.

``docs/legacy-removal.md`` was the plan Steps 11 to 13 worked from: every
endpoint the Flask-era surface served, who called it, what answers it on the
contract, and which wave removed it. While the plan was live this module held
it against the real routing table in both directions, so it could not promise
work that was done or leave a route nobody had decided about.

All three waves are done. The plan is a record now, and what this module holds
is the other end of the same rule: **nothing on the page is served any more,
and nothing in the tree still reaches for it.** A removal that is only a
deletion comes back -- as a re-added route, a dependency nobody dropped, a
launcher still starting a process that is gone -- and each of those is a line
below.

What is deliberately *not* checked here is the caller column of the record. It
was evidence gathered by reading two repositories and a container file; a test
that re-derived it would be re-deriving it wrongly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import asgi as entrypoint
import interfaces.http as http

REPO = Path(__file__).resolve().parents[2]
MAP = REPO / "docs" / "legacy-removal.md"
V1 = http.v1.PREFIX

#: Rows of the removal record: ``| VERB /path | replacement | ... |``.
ROW = re.compile(r"^\|\s*`([A-Z]+)\s+(/api/[^`]+)`\s*\|\s*(.*?)\s*\|", re.M)
#: A replacement cell either names one `/api/v1` endpoint or is an em dash.
REPLACEMENT = re.compile(r"`([A-Z]+)\s+(/api/v1/[^`]+)`")

#: Everything wave 3 deleted, by path. A file back in the tree is the plainest
#: way for this removal to be undone.
DELETED = (
    "app.py", "wsgi.py",
    "interfaces/http/legacy", "interfaces/http/coexistence.py",
    "interfaces/http/context.py",
    "templates", "static",
)

#: Dependencies that went with the surface they served.
DROPPED_PACKAGES = ("flask", "flask-cors", "waitress")

#: What no request may still be addressed to, and what answers it instead.
#: Checked against *string literals* rather than against the text of a file:
#: a comment or a docstring that records the removal is not a reference to it,
#: and this repository is full of both on purpose.
GONE = {
    "/api/demo/": "the Viewer relay; the Viewer is a screen over /api/v1",
    "/api/kb": "GET|POST /api/v1/knowledge-bases",
    "/api/documents/upload": "POST /api/v1/documents",
    "/api/ingest/jobs": "/api/v1/ingest-jobs",
    "/api/experiment/": "POST /api/v1/searches",
    "/api/goldset": "the gold set is an offline input to `python -m cli`",
    "/api/chunks": "POST /api/v1/searches, or a knowledge base's own chunks",
    "/api/query": "POST /api/v1/queries",
    "/api/stats": "GET /api/v1/documents carries every number that was in it",
    "/api/health": "GET /api/v1/health",
}

#: Where a live reference would actually do damage. Python is read as an
#: abstract syntax tree; the front end's own guard is exempt because listing
#: the forbidden paths *is* what it does (``frontend/tests/surface.test.ts``).
FRONTEND = ("frontend/app", "frontend/components", "frontend/lib", "frontend/types")
EXEMPT_FRONTEND = {"frontend/tests/surface.test.ts"}


#: This file names every removed spelling, in the table above.
SELF = Path(__file__).name


def _python_sources() -> list[Path]:
    found = []
    for path in REPO.rglob("*.py"):
        relative = path.relative_to(REPO).as_posix()
        if relative.startswith((".git/", "frontend/", "venv/", ".venv/",
                               "artifacts/", ".cache/", "tests/fixtures/")):
            continue
        if path.name == SELF:
            continue
        found.append(path)
    return found


def _frontend_sources() -> list[Path]:
    found = []
    for prefix in FRONTEND:
        base = REPO / prefix
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".ts", ".tsx", ".mjs"}:
                if path.relative_to(REPO).as_posix() not in EXEMPT_FRONTEND:
                    found.append(path)
    return found


def _string_literals(source: str):
    """Every string constant in a Python module that is not a docstring.

    A docstring is prose: this file's own subject is a removal, and half the
    modules in this repository explain it in one. What matters is a literal
    something could still *send*.
    """
    import ast

    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            yield node.lineno, node.value


def _mapped() -> dict[tuple[str, str], str]:
    """Every endpoint the record classifies, and what it says replaced it."""
    rows = {(verb, path): replacement
            for verb, path, replacement in ROW.findall(MAP.read_text(encoding="utf-8"))}
    assert rows, "the removal record classifies no endpoint"
    return rows


# ------------------------------------------------------------- it happened
def test_nothing_the_record_classifies_is_served_any_more():
    """The whole point of the page, in one line. Every row on it was a route;
    none of them is in the routing table."""
    served = http.surface(entrypoint.application)
    still_here = sorted(row for row in _mapped() if row in served)
    assert still_here == [], (
        "these were planned for removal and are still served: " + repr(still_here))


def test_the_only_thing_served_outside_the_contract_is_the_operator_route():
    """``tests/migration/test_http_surface.py`` publishes that one. Anything
    else outside ``/api/v1`` would be a legacy surface growing back."""
    outside = {row for row in http.surface(entrypoint.application)
               if not row[1].startswith(f"{V1}/")}
    assert outside == {("GET", "/api/ops/metrics")}, sorted(outside)


@pytest.mark.parametrize("relative", DELETED)
def test_the_files_wave_three_deleted_are_gone(relative):
    """A framework comes back one file at a time. These are the files."""
    assert not (REPO / relative).exists(), f"{relative} is back in the tree"


def test_no_module_imports_the_framework_that_was_removed():
    """The strongest statement of "the console is not Flask": nothing imports
    it, including the tests, so a route cannot quietly be added to it."""
    offenders = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^\s*(import flask|from flask)", text, re.M):
            offenders.append(path.relative_to(REPO).as_posix())
    assert offenders == [], offenders


@pytest.mark.parametrize("package", DROPPED_PACKAGES)
def test_the_dependencies_that_went_with_it_are_not_declared(package):
    """A dependency left in ``requirements.txt`` is one a fresh image installs
    and a reader believes is used."""
    declared = [
        line.split("#", 1)[0].strip().lower()
        for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert package not in declared, f"{package} is still declared"


@pytest.mark.parametrize("spelling,instead", sorted(GONE.items()))
def test_no_source_still_addresses_a_removed_route(spelling, instead):
    """A stale path in a string literal is not a documentation problem -- it is
    a request that will 404."""
    offenders = []
    for path in _python_sources():
        for number, literal in _string_literals(
                path.read_text(encoding="utf-8", errors="replace")):
            if spelling in literal and "/api/v1" not in literal:
                offenders.append(f"{path.relative_to(REPO).as_posix()}:{number}")
    for path in _frontend_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if spelling in line and "/api/v1" not in line:
                offenders.append(f"{path.relative_to(REPO).as_posix()}:{number}")
    assert offenders == [], (
        f"{spelling!r} is gone; use {instead}. Still addressed at: {offenders}")


def test_nothing_starts_a_second_server_any_more():
    """The launchers are the other way this comes back: a script that starts
    the Flask console, or the Viewer process on ``:8765``, would put both back
    into a demo without a line of Python changing."""
    for script in ("start-demo.ps1", "stop-demo.ps1"):
        text = (REPO / script).read_text(encoding="utf-8")
        commands = [line for line in text.splitlines()
                    if "Start-Process" in line or "ArgumentList" in line
                    or "Signatures" in line]
        joined = "\n".join(commands)
        assert "app.py" not in joined, f"{script} still starts the Flask console"
        assert "wsgi" not in joined, f"{script} still starts the WSGI server"
        assert "8765" not in joined, f"{script} still touches the Viewer process"


def test_the_companion_viewer_setting_is_not_read_anywhere():
    """``VIEWER_URL`` pointed at the process on ``:8765``. There is no such
    process, so there is nothing for the setting to mean."""
    # ``tests/unit/test_documentation.py`` names it in the list of settings the
    # documents may mention *because* they are gone, which is the opposite of
    # reading it.
    exempt = {"tests/unit/test_documentation.py"}
    offenders = []
    for path in _python_sources():
        relative = path.relative_to(REPO).as_posix()
        if relative in exempt:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if "VIEWER_URL" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{relative}:{number}")
    assert offenders == [], offenders


# ----------------------------------------------------------- it is recorded
def test_the_record_still_names_a_replacement_the_contract_serves():
    """The rows are history now, and history that names an endpoint nobody
    serves is history a reader cannot follow."""
    served = http.surface(entrypoint.application)
    missing = []
    for (verb, path), replacement in _mapped().items():
        for answer in REPLACEMENT.findall(replacement):
            if answer not in served:
                missing.append(f"{verb} {path} -> {answer[0]} {answer[1]}")
    assert missing == [], missing


def test_an_endpoint_with_no_replacement_says_so_rather_than_leaving_the_cell_blank():
    """`—` is a decision to stop offering something, and the reason is in
    ``docs/api-v1.md``. An empty cell is an unanswered question."""
    unexplained = [f"{verb} {path}" for (verb, path), replacement in _mapped().items()
                   if not replacement.strip()]
    assert unexplained == [], unexplained


@pytest.mark.parametrize("wave", ["## Wave 1", "## Wave 2", "## Wave 3"])
def test_all_three_waves_are_still_on_the_record(wave):
    """The order was the plan, and it is why the removal did not break three
    things at once. A record that drops a wave loses that."""
    assert wave in MAP.read_text(encoding="utf-8")


def test_the_readme_no_longer_advertises_a_console_api():
    """The table that published the Flask-era surface is gone with it. A
    reader arriving at the README is offered one API."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "## The console API" not in readme
    assert "## The product API" in readme
