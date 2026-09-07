"""The pin-promotion command: it moves the sha, and it refuses to move it wrongly.

`chat_rag` installs the chunking library from an immutable commit. Promoting a
library change is therefore a release step, and the step is mechanical enough
that a human doing it by hand is the risk: `test_amsc_pin.py` exists because a
sha was once copied short and a newline once went missing. `tools/promote_chunk_pin.py`
does the copying.

What is tested here is the part a mistake hides in -- the refusals -- and the
one guarantee that makes the tool safe to reach for: it rewrites the sha of
the one requirement line and touches nothing else in the file. Each case runs
against a throwaway git repository built in ``tmp_path``, so no network, no
sibling checkout and no state of the developer's is involved.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import promote_chunk_pin as promote  # noqa: E402

REQUIREMENTS_BEFORE = """python-dotenv
sentence-transformers
# a comment about the pin
amsc-poc @ git+https://github.com/erenayd58/chunk.git@{sha}
openai
flask
"""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(repo), *args],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture
def library(tmp_path):
    """A tiny git repository with a remote, standing in for ``chunk``.

    A real remote (a bare repository on disk) rather than a fake one, because
    the check that matters -- "is this commit fetchable by anybody else" -- is
    answered from the remote-tracking refs.
    """
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   capture_output=True, check=True)
    repo = tmp_path / "chunk"
    subprocess.run(["git", "init", "-b", "main", str(repo)], capture_output=True, check=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "remote", "add", "origin", str(origin))
    (repo / "src").mkdir()
    (repo / "src" / "thing.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "first")
    _git(repo, "push", "-u", "origin", "main")
    return repo


@pytest.fixture
def requirements(tmp_path, monkeypatch):
    """A requirements.txt of our own, so no test can rewrite the real one."""
    path = tmp_path / "requirements.txt"
    path.write_text(REQUIREMENTS_BEFORE.format(sha="0" * 40), encoding="utf-8")
    monkeypatch.setattr(promote, "REQUIREMENTS", path)
    monkeypatch.setattr(promote, "validate", lambda python: subprocess.CompletedProcess([], 0, "", ""))
    return path


def head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def run(library: Path, *extra: str) -> int:
    return promote.main(["--chunk-repo", str(library), *extra])


# ------------------------------------------------------------ it promotes


def test_it_writes_the_new_sha_and_changes_nothing_else(library, requirements, capsys):
    assert run(library) == 0
    after = requirements.read_text(encoding="utf-8")
    assert after == REQUIREMENTS_BEFORE.format(sha=head(library))
    assert "wrote" in capsys.readouterr().out


def test_promoting_the_same_commit_twice_is_a_no_op(library, requirements, capsys):
    assert run(library) == 0
    written = requirements.read_text(encoding="utf-8")
    assert run(library) == 0
    assert requirements.read_text(encoding="utf-8") == written
    assert "already" in capsys.readouterr().out


def test_check_reports_the_move_and_writes_nothing(library, requirements, capsys):
    before = requirements.read_text(encoding="utf-8")
    assert run(library, "--check") == 0
    assert requirements.read_text(encoding="utf-8") == before
    out = capsys.readouterr().out
    assert head(library)[:7] in out and "nothing written" in out


def test_an_explicit_revision_is_pinned_rather_than_head(library, requirements):
    first = head(library)
    (library / "src" / "thing.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(library, "commit", "-am", "second")
    _git(library, "push", "origin", "main")
    assert run(library, "--rev", first[:12]) == 0
    assert first in requirements.read_text(encoding="utf-8")


# ------------------------------------------------------------- it refuses


def test_it_refuses_a_commit_no_remote_branch_contains(library, requirements, capsys):
    """The failure a clean install hits and a developer machine never does."""
    (library / "src" / "thing.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(library, "commit", "-am", "unpushed")
    assert run(library) == 1
    assert "on no remote branch" in capsys.readouterr().err
    assert "0" * 40 in requirements.read_text(encoding="utf-8"), "the pin is untouched"


def test_it_refuses_while_the_library_has_uncommitted_changes(library, requirements, capsys):
    (library / "src" / "thing.py").write_text("VALUE = 4\n", encoding="utf-8")
    assert run(library) == 1
    assert "HEAD does not contain" in capsys.readouterr().err


def test_it_refuses_while_a_new_file_is_untracked(library, requirements, capsys):
    """The one a modified-files check misses: a new chunker is a new file."""
    (library / "src" / "new_chunker.py").write_text("# a fifth method\n", encoding="utf-8")
    assert run(library) == 1
    error = capsys.readouterr().err
    assert "HEAD does not contain" in error and "new_chunker.py" in error


def test_it_refuses_a_revision_that_is_not_a_commit(library, requirements, capsys):
    assert run(library, "--rev", "no-such-thing") == 1
    assert "not a commit" in capsys.readouterr().err


def test_it_refuses_when_there_is_no_library_checkout(requirements, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("CHUNK_REPO", raising=False)
    monkeypatch.setattr(promote, "ROOT", tmp_path / "nowhere")
    assert promote.main([]) == 1
    assert "no chunk checkout found" in capsys.readouterr().err


def test_a_failing_pin_test_puts_requirements_back(library, requirements, monkeypatch, capsys):
    """A promotion that does not hold is not a promotion.

    The usual cause is real: the product imports a symbol the revision does
    not have yet. Leaving the file rewritten would mean the next command in
    the release sequence fails against a file somebody has to remember to
    revert.
    """
    before = requirements.read_text(encoding="utf-8")
    monkeypatch.setattr(
        promote, "validate",
        lambda python: subprocess.CompletedProcess([], 1, "E   assert missing == []\n", ""),
    )
    assert run(library) == 1
    assert requirements.read_text(encoding="utf-8") == before
    assert "put back" in capsys.readouterr().err


# ------------------------------------------------------- the line it edits


def test_only_the_sha_of_the_one_pin_line_moves():
    text = REQUIREMENTS_BEFORE.format(sha="a" * 40)
    rewritten = promote.rewrite(text, "b" * 40)
    assert rewritten == REQUIREMENTS_BEFORE.format(sha="b" * 40)
    assert rewritten.count("b" * 40) == 1
    assert promote.current_pin(rewritten) == "b" * 40
    # The line still ends at the sha -- the shape test_amsc_pin.py checks.
    line = next(l for l in rewritten.splitlines() if l.startswith("amsc-poc"))
    assert line.rsplit("@", 1)[-1] == "b" * 40


def test_a_requirements_file_with_no_pin_line_is_an_error():
    with pytest.raises(promote.Failure, match="no `amsc-poc"):
        promote.current_pin("flask\nopenai\n")


def test_the_real_requirements_file_is_one_this_tool_can_read():
    """The tool and ``test_amsc_pin.py`` must agree about the real file."""
    text = promote.REQUIREMENTS.read_text(encoding="utf-8")
    assert len(promote.current_pin(text)) == 40
