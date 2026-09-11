"""Promote the sibling ``chunk`` checkout's commit into both pins.

`chat_rag` installs the chunking library from an **immutable commit**, which
is the right dependency model and a tedious release step: commit in `chunk`,
push, copy a forty-character sha out of `git log`, paste it into
`requirements.txt` *and* `pyproject.toml`, run the pin tests. The sha is the
part a human should never be moving by hand -- a typo, a truncation or a lost
newline all fail somewhere later and less clearly (`test_amsc_pin.py` exists
because two of those three happened), and two copies of it are two chances.

Two files name the commit because two things install it: `requirements.txt`
is the repository's own install and the image's, and `pyproject.toml` is the
wheel's metadata -- the direct reference that lets a consumer's `pip install`
resolve `amsc-poc` with neither checkout present. They are held equal by
`test_amsc_pin.py`, and this tool is what moves them together.

    python tools/promote_chunk_pin.py                # HEAD of ../chunk
    python tools/promote_chunk_pin.py --rev abc1234  # a particular revision
    python tools/promote_chunk_pin.py --check        # report, change nothing

What it does, in this order, stopping at the first failure:

1. find the `chunk` checkout (``--chunk-repo``, ``CHUNK_REPO``, then
   ``../chunk``) and resolve the revision to a full sha;
2. refuse a revision with **uncommitted changes in the working tree** at
   HEAD -- pinning a commit that does not contain what you just edited is the
   mistake this whole model exists to prevent;
3. refuse a revision **no remote branch contains**: a clean install cannot
   fetch it, and it will pass on this machine and on no other. ``--push``
   pushes the checkout's current branch to its upstream first, which is the
   only network call this tool ever makes and only when asked;
4. rewrite the one ``amsc-poc @ git+...@<sha>`` requirement in each file,
   leaving the URL, the rest of the line and every other line exactly as they
   were;
5. run the pin tests (``tests/unit/test_amsc_pin.py``), and restore both
   files if they fail, so a bad promotion never survives the command.

It does not commit, tag, or push this repository: it prints the `git` command
for the change it made and leaves the decision to the developer. Nothing else
in the release sequence is automated, because nothing else in it is
mechanical.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"

#: The requirement line, split so the sha can be replaced and nothing else.
PIN_LINE = re.compile(r"^(?P<head>amsc-poc\s*@\s*\S+?@)(?P<sha>[0-9a-f]{7,40})(?P<tail>\s*)$", re.M)
#: The same requirement as ``pyproject.toml`` writes it: one quoted entry of
#: the ``dependencies`` list. The quotes are the anchors.
WHEEL_PIN = re.compile(r'(?P<head>"amsc-poc\s*@\s*\S+?@)(?P<sha>[0-9a-f]{7,40})(?P<tail>")')


class Failure(Exception):
    """Something the developer has to fix; printed without a traceback."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8"
    )


def find_chunk_repo(given: str | None = None) -> Path:
    for candidate in (given, os.environ.get("CHUNK_REPO"), str(ROOT.parent / "chunk")):
        if candidate and (Path(candidate) / ".git").exists():
            return Path(candidate).resolve()
    raise Failure(
        "no chunk checkout found. Pass --chunk-repo, set CHUNK_REPO, or put the "
        f"repository beside this one at {ROOT.parent / 'chunk'}"
    )


def resolve(repo: Path, revision: str) -> str:
    found = _git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if found.returncode != 0:
        raise Failure(f"{revision!r} is not a commit in {repo}")
    return found.stdout.strip()


def dirty(repo: Path) -> list[str]:
    """What HEAD does not contain: modified tracked files **and untracked new
    ones**. The second half matters more than it looks -- a new chunker is a
    new file, and a file nobody ran ``git add`` on is invisible to a
    modified-files check while being exactly what the pin was meant to carry.
    """
    status = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    return [line.rstrip() for line in status.stdout.splitlines() if line.strip()]


def remote_branches(repo: Path, commit: str) -> list[str]:
    """The remote branches that contain the commit, so a clean install can fetch it."""
    found = _git(repo, "branch", "-r", "--contains", commit)
    if found.returncode != 0:
        return []
    return [line.strip() for line in found.stdout.splitlines() if line.strip()]


def push(repo: Path) -> str:
    """Push the checkout's current branch to its upstream. The one network call."""
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if not branch or branch == "HEAD":
        raise Failure("the chunk checkout is on no branch; cannot push a detached HEAD")
    pushed = _git(repo, "push", "origin", branch)
    if pushed.returncode != 0:
        raise Failure(f"git push origin {branch} failed:\n{pushed.stderr.strip()}")
    return branch


def current_pin(text: str) -> str:
    match = PIN_LINE.search(text)
    if not match:
        raise Failure(
            "requirements.txt has no `amsc-poc @ git+<url>@<sha>` line on a line "
            "of its own; fix it by hand before promoting a pin"
        )
    return match.group("sha")


def rewrite(text: str, commit: str) -> str:
    """Replace only the sha of the one pin line."""
    return PIN_LINE.sub(lambda m: m.group("head") + commit + m.group("tail"), text, count=1)


def current_wheel_pin(text: str) -> str:
    match = WHEEL_PIN.search(text)
    if not match:
        raise Failure(
            'pyproject.toml has no `"amsc-poc @ git+<url>@<sha>"` dependency; '
            "fix it by hand before promoting a pin"
        )
    return match.group("sha")


def rewrite_wheel(text: str, commit: str) -> str:
    """Replace only the sha of the one dependency entry."""
    return WHEEL_PIN.sub(lambda m: m.group("head") + commit + m.group("tail"), text, count=1)


def validate(python: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [python, "-m", "pytest", "-q", "tests/unit/test_amsc_pin.py"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/promote_chunk_pin.py",
        description="Point requirements.txt at a chunk commit, and check it holds",
    )
    parser.add_argument("--chunk-repo", help="the chunk checkout (default: CHUNK_REPO, then ../chunk)")
    parser.add_argument("--rev", default="HEAD", help="the revision to pin (default: HEAD)")
    parser.add_argument("--push", action="store_true",
                        help="push the chunk checkout's current branch first")
    parser.add_argument("--check", action="store_true",
                        help="report what would change and write nothing")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip running tests/unit/test_amsc_pin.py afterwards")
    parser.add_argument("--python", default=sys.executable,
                        help="the interpreter the pin tests run under")
    args = parser.parse_args(argv)

    try:
        repo = find_chunk_repo(args.chunk_repo)
        print(f"chunk repo    {repo}")

        if args.push and not args.check:
            print(f"pushed        origin/{push(repo)}")

        commit = resolve(repo, args.rev)
        subject = _git(repo, "log", "-1", "--format=%s", commit).stdout.strip()
        print(f"revision      {commit}  ({args.rev})\n              {subject}")

        if args.rev == "HEAD":
            changed = dirty(repo)
            if changed:
                more = f"\n  ... and {len(changed) - 10} more" if len(changed) > 10 else ""
                raise Failure(
                    "the chunk checkout has changes HEAD does not contain "
                    "(a `??` line is a file nobody ran `git add` on):\n  "
                    + "\n  ".join(changed[:10]) + more
                    + "\ncommit them first, or pin an explicit --rev"
                )

        branches = remote_branches(repo, commit)
        if not branches:
            raise Failure(
                f"{commit[:7]} is on no remote branch, so a clean install cannot "
                "fetch it. Push it (or re-run with --push)."
            )
        print(f"on remote     {', '.join(branches)}")

        text = REQUIREMENTS.read_text(encoding="utf-8")
        wheel_text = PYPROJECT.read_text(encoding="utf-8")
        before = current_pin(text)
        wheel_before = current_wheel_pin(wheel_text)
        if before == commit and wheel_before == commit:
            print(f"pin           already {commit[:7]}; nothing to change")
            return 0
        print(f"pin           {before[:7]} -> {commit[:7]}  ({REQUIREMENTS.name})")
        print(f"              {wheel_before[:7]} -> {commit[:7]}  ({PYPROJECT.name})")
        if args.check:
            print("check         nothing written (--check)")
            return 0

        REQUIREMENTS.write_text(rewrite(text, commit), encoding="utf-8", newline="\n")
        PYPROJECT.write_text(rewrite_wheel(wheel_text, commit), encoding="utf-8", newline="\n")
        print(f"wrote         {REQUIREMENTS.name}, {PYPROJECT.name}")

        if not args.no_validate:
            result = validate(args.python)
            if result.returncode != 0:
                REQUIREMENTS.write_text(text, encoding="utf-8", newline="\n")
                PYPROJECT.write_text(wheel_text, encoding="utf-8", newline="\n")
                sys.stdout.write(result.stdout)
                sys.stderr.write(result.stderr)
                raise Failure(
                    f"the pin tests fail against {commit[:7]}; requirements.txt and "
                    "pyproject.toml have been put back. Usually this means the "
                    "product imports an amsc symbol that revision does not have yet."
                )
            print("validated     tests/unit/test_amsc_pin.py")

        print(
            "\nnext          git -C . add requirements.txt pyproject.toml && "
            f'git -C . commit -m "build: pin amsc-poc to {commit[:7]}"'
        )
        return 0
    except Failure as failure:
        print(f"\nFAILED: {failure}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
