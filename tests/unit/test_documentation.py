"""The documentation is checked, not just written.

A handover package is only worth the trust a new developer puts in it, and the
things that quietly rot are always the same four: a link to a file somebody
deleted, a command with a previous author's home directory in it, a commit sha
copied into prose that has since moved on, and a setting the docs still
describe after the code stopped reading it.

Each of those is mechanical to check, so it is checked here rather than
re-noticed by whoever reads the docs next.

The env-name half is the same rule ``test_configuration.py`` applies to
``env.example``, extended to prose: a variable a document tells you to set must
be one the application actually reads. Naming a *removed* setting is legitimate
— ``limitations.md`` explains why there is no upload size cap by naming the
knob that used to pretend there was — so those are listed, with the reason.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHUNK = REPO.parent / "chunk"

#: Env-var-shaped names a document may mention although no code reads them,
#: because the document is explaining that they are gone.
DOCUMENTED_AS_REMOVED = {
    "MAX_FILE_SIZE_MB": "limitations.md — why there is no upload size cap",
    "LLM_MAX_TOKENS": "configuration.md — an example of a knob that turned nothing",
    "PDF_PARSER_BACKEND": "configuration.md — the same",
    "OCR_LANGUAGE": "configuration.md — the same",
    "LOG_TOKEN_USAGE": "configuration.md — the setting Phase 7B found inert",
}

#: Names shaped like a setting that are really Python constants in the
#: *library*, which this repository documents but does not define. The scan
#: below reads only this repository's own tracked ``*.py``, so a constant that
#: lives in ``chunk`` can never satisfy it however correctly it is documented
#: — the boundary the docs describe is exactly the boundary that hides it.
#:
#: Each entry names the file it is defined in, and
#: ``test_every_library_constant_listed_really_exists`` checks that against the
#: sibling checkout, so this is a statement about the library that can be
#: falsified rather than a way to silence the guard.
LIBRARY_CONSTANTS = {
    "CONSOLE_API": "src/amsc/surface.py — the modules chat_rag may import",
}

#: Files whose *content* is the source of truth for a variable name.
_SOURCE_GLOBS = ("*.py", "*.ps1", "*.yml", "*.yaml", "Dockerfile", "env.example",
                 ".env.docker", "requirements.txt")


def _tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", *patterns], cwd=REPO,
                         capture_output=True, text=True).stdout.split()
    return [REPO / name for name in out]


def _shipping_source() -> str:
    """Everything that runs in production, and nothing that only tests it.

    A test naming a setting is not the application reading it -- that is the
    same distinction ``test_configuration.py`` draws, and it is what stops a
    docstring about a removed knob from looking like the knob coming back.
    """
    return "\n".join(
        _read(path)
        for pattern in _SOURCE_GLOBS
        for path in _tracked(pattern)
        if "tests" not in path.relative_to(REPO).parts
    )


def _markdown() -> list[Path]:
    """Every document a reader is meant to follow.

    Frozen evaluation reports and fixture notes are records of a run, not
    instructions, and are left alone.
    """
    skip = {"artifacts", "evaluation", "tests", "venv", "node_modules"}
    return [
        path for path in _tracked("*.md")
        if not (set(path.relative_to(REPO).parts[:-1]) & skip)
    ]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------- links


def test_every_relative_link_in_the_docs_resolves():
    """A link to a file that is not there is worse than no link."""
    broken = []
    for path in _markdown():
        for match in re.finditer(r"\[([^\]]+)\]\(([^)\s]+)\)", _read(path)):
            target = match.group(2).split("#")[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if resolved.is_relative_to(CHUNK) and not CHUNK.exists():
                continue  # the sibling library is not checked out here
            if not resolved.exists():
                broken.append(f"{path.relative_to(REPO).as_posix()} -> {target}")
    assert broken == [], "documentation links that do not resolve:\n  " + "\n  ".join(broken)


def test_the_cross_repository_links_point_at_files_that_exist():
    """The handover spans two repositories; the links between them are the
    part a reader cannot repair by guessing."""
    if not CHUNK.exists():
        pytest.skip("no chunk checkout beside this one")
    expected = [
        CHUNK / "docs" / "adding-a-chunker.md",
        CHUNK / "docs" / "viewer-architecture.md",
        CHUNK / "docs" / "library-surface.md",
        CHUNK / "docs" / "package-layout.md",
        CHUNK / "src" / "amsc" / "chunking" / "example.py",
        CHUNK / "src" / "amsc" / "chunking" / "registry.py",
        CHUNK / "src" / "amsc" / "surface.py",
    ]
    missing = [p.name for p in expected if not p.exists()]
    assert missing == [], f"the docs point at library files that are gone: {missing}"


# ----------------------------------------------------------- machine paths


def test_nothing_tracked_names_a_developers_home_directory():
    """A path out of somebody's checkout is noise at best and a wrong
    instruction at worst. Fifty-eight files carried one as a header comment
    until Phase 9."""
    pattern = re.compile(r"(/Users/[a-z]|/home/[a-z]|C:\\Users\\[A-Za-z])")
    offenders = []
    for path in _tracked():
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".ico", ".pyc"}:
            continue
        try:
            text = _read(path)
        except OSError:  # pragma: no cover - unreadable blob
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line) and "example.com" not in line:
                offenders.append(f"{path.relative_to(REPO).as_posix()}:{number}")
    assert offenders == [], (
        "these name a developer's own machine:\n  " + "\n  ".join(offenders)
    )


# ------------------------------------------------------------- stale shas


def test_no_document_writes_a_commit_sha():
    """``requirements.txt`` is the one place a revision is written.

    The README used to name the pinned commit twice in prose. Both copies had
    drifted from the pin by the time anyone looked, which is the failure mode
    of writing a value in two places.
    """
    sha = re.compile(r"\b[0-9a-f]{40}\b|\b[0-9a-f]{7,12}\b")
    offenders = []
    for path in _markdown():
        for number, line in enumerate(_read(path).splitlines(), 1):
            if "sha256" in line.lower() or "hash" in line.lower():
                continue
            for candidate in sha.findall(line):
                if candidate.isdigit():
                    continue  # a number, not a revision
                offenders.append(
                    f"{path.relative_to(REPO).as_posix()}:{number}: {candidate}")
    assert offenders == [], (
        "documents naming a commit revision; requirements.txt owns the pin:\n  "
        + "\n  ".join(offenders)
    )


# -------------------------------------------------------------- env names


def test_every_setting_the_docs_name_is_one_the_code_reads():
    """A document that tells you to set something must be right about it."""
    source = _shipping_source()
    filenames = {path.stem for path in _tracked()}
    # A doc may also name a Python constant -- ``DELIBERATE_OVERRIDES`` is a
    # table in a test, not a setting. Those are defined, not read from the
    # environment, so they are told apart by being assigned somewhere.
    constants = {
        match.group(1)
        for path in _tracked("*.py")
        for match in re.finditer(r"^([A-Z][A-Z0-9_]+)\s*[:=]", _read(path), re.M)
    }
    named: dict[str, set[str]] = {}
    for path in _markdown():
        for match in re.finditer(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b", _read(path)):
            named.setdefault(match.group(1), set()).add(
                path.relative_to(REPO).as_posix())

    unread = {
        name: sorted(where)
        for name, where in sorted(named.items())
        if name not in source
        and name not in DOCUMENTED_AS_REMOVED
        and name not in LIBRARY_CONSTANTS
        and name not in filenames
        and name not in constants
    }
    assert unread == {}, "\n".join(
        ["documents name settings no code reads; wire them, remove them, or "
         "list them in DOCUMENTED_AS_REMOVED (a knob that is gone) or "
         "LIBRARY_CONSTANTS (a constant that lives in chunk), with the reason:"]
        + [f"  {name}  <- {', '.join(where)}" for name, where in unread.items()]
    )


def test_every_removal_listed_is_really_removed():
    """The allow-list must not outlive the removals it explains."""
    source = _shipping_source()
    resurrected = sorted(name for name in DOCUMENTED_AS_REMOVED if name in source)
    assert resurrected == [], (
        f"these are read by code again; drop them from DOCUMENTED_AS_REMOVED: "
        f"{resurrected}"
    )


def test_every_library_constant_listed_really_exists():
    """The other allow-list, held to the same standard.

    An entry claims two things -- that the name is defined in the library, and
    where -- and both are checked against the sibling checkout. Without one the
    test skips rather than guessing, exactly as ``test_amsc_pin.py`` does; the
    claim is then unverified here but it is never quietly assumed true.
    """
    if not CHUNK.exists():
        pytest.skip("no chunk checkout beside this one")
    wrong = []
    for name, where in sorted(LIBRARY_CONSTANTS.items()):
        path = CHUNK / where.split(" — ")[0].split(" -- ")[0].strip()
        if not path.is_file():
            wrong.append(f"{name}: {path.name} does not exist in the library")
            continue
        if not re.search(rf"^{re.escape(name)}\s*[:=]", _read(path), re.M):
            wrong.append(f"{name} is not defined in {path.name}")
    assert wrong == [], (
        "LIBRARY_CONSTANTS describes the library wrongly:\n  " + "\n  ".join(wrong)
    )


def test_no_library_constant_hides_a_setting_this_repository_reads():
    """The allow-list must not be a way to stop checking a real setting.

    If a name on it ever becomes something the application reads from the
    environment, the entry is wrong and the ordinary rule should apply again.
    """
    source = _shipping_source()
    shadowed = sorted(
        name for name in LIBRARY_CONSTANTS
        if re.search(rf"getenv\(\s*[\"']{re.escape(name)}[\"']", source)
        or re.search(rf"environ\[\s*[\"']{re.escape(name)}[\"']", source)
    )
    assert shadowed == [], (
        "these are read from the environment here after all; drop them from "
        f"LIBRARY_CONSTANTS: {shadowed}"
    )


# ------------------------------------------------------- the entry points


def test_the_documented_entry_points_exist():
    """Every doc the README sends a new developer to."""
    for relative in ("docs/api-v1.md", "docs/architecture.md", "docs/operations.md",
                     "docs/configuration.md", "docs/testing.md",
                     "docs/limitations.md", "env.example",
                     "tools/import_smoke.py", "tools/serve_smoke.py",
                     "tools/verify_reproducibility.py",
                     "tools/promote_chunk_pin.py"):
        assert (REPO / relative).exists(), relative


def test_every_document_the_readme_links_to_travels_with_a_clone():
    """``docs/`` is git-ignored by default, with an allow-list.

    That is the right default -- the directory is otherwise scratch and
    generated output -- and it is also a trap: a new document is invisible to
    everyone but its author until `.gitignore` names it, and nothing else in
    the suite would notice. A handover doc that is not in the clone is worse
    than no doc, because the README promises it.
    """
    readme = _read(REPO / "README.md")
    linked = sorted({
        match.group(2).split("#")[0]
        for match in re.finditer(r"\[([^\]]+)\]\(([^)\s]+)\)", readme)
        if match.group(2).startswith("docs/")
    })
    assert linked, "the README links to no document at all"

    # `git check-ignore` answers the question that matters -- would a normal
    # `git add` pick this up -- for a file that is merely new as well as one
    # that is already tracked.
    ignored = subprocess.run(
        ["git", "check-ignore", *linked], cwd=REPO,
        capture_output=True, text=True,
    ).stdout.split()
    assert ignored == [], (
        "the README links to these but .gitignore excludes them, so they would "
        f"not travel with a clone; add each to the allow-list: {ignored}"
    )


def test_the_readme_is_an_entry_point_not_a_manual():
    """It was 1387 lines, half of them a copy of something else.

    Not a style rule: a README nobody finishes is a README whose stale half
    goes unnoticed, which is how it came to promise two guides that were never
    written and a Python version the library refuses.
    """
    readme = _read(REPO / "README.md")
    assert len(readme.splitlines()) < 800, "the README is growing back into a manual"
    for doc in ("docs/api-v1.md", "docs/architecture.md", "docs/operations.md",
                "docs/configuration.md", "docs/testing.md", "docs/limitations.md"):
        assert doc in readme, f"the README no longer points at {doc}"
