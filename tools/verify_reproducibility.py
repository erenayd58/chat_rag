"""Prove that a clean clone of this repository installs, starts and answers.

    python tools/verify_reproducibility.py               # the default gate
    python tools/verify_reproducibility.py --with-host-install
    python tools/verify_reproducibility.py --no-docker --keep

Why this exists
---------------

A repository can pass five hundred tests while being impossible to install.
That is not hypothetical here: one dependency line lost its newline, which
merged a package name into a git revision, and ``pip install -r
requirements.txt`` failed outright for a whole phase. Every suite stayed green,
because a developer checkout already had the package and used an editable
install of the sibling library. Tests prove the code is right. This proves the
*declared source* is enough -- which is a different claim, and the one another
engineer depends on.

So every check here runs against material fetched from the remote, in an
environment built from nothing, and each one is written to fail if it could
only have passed because of this machine: an editable sibling checkout, a
globally installed package, a developer ``.env``, a warm model cache, existing
Chroma data or leftovers from a previous run.

What it reports
---------------

Each check is PASS, FAIL or SKIP. SKIP means a capability is unavailable (no
Docker, no Python 3.11, no network) or a tier was not asked for -- never that
something was checked and forgiven. The exit code is non-zero only on FAIL.

Tiers
-----

``chunk`` and ``clone`` always run; they are quick and they cover the pinned
dependency, the Viewer shell and the pushed-ness of this branch.

``docker`` runs when Docker is available. It is the primary clean-environment
proof: ``python:3.11-slim`` has no editable siblings, no developer packages and
no caches, and the image build already runs the import and serve smokes inside
itself.

``--with-host-install`` adds the same install on *this* machine, outside a
container. It is off by default because it downloads a gigabyte or so of
wheels and takes several minutes; when it is off, the line says so rather than
quietly reporting success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

#: Files in the developer's checkout that no check may touch. Fingerprinted
#: before and after, because "used an isolated data root" is a claim, and this
#: is the evidence for it.
DEVELOPER_STATE = (
    ".ingested_documents.json",
    ".knowledge_bases.json",
    ".gold_set.json",
    "chroma_db/chroma.sqlite3",
)

#: Things a clean clone must not contain. Each is real state or real output
#: from this machine; a clone that carried one would let a check pass for the
#: wrong reason.
MUST_NOT_BE_CLONED = (
    ".env",
    ".knowledge_bases.json",
    ".ingested_documents.json",
    ".gold_set.json",
    "chroma_db",
    ".cache",
    "venv",
    ".venv",
    "artifacts/viewer-live",
    "logs",
)


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.details: dict[str, str] = {}

    def add(self, status: str, name: str, detail: str, output: str = "") -> str:
        self.rows.append((status, name, detail))
        mark = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP"}[status]
        print(f"  {mark}  {name:<22} {detail}", flush=True)
        if output and status == FAIL:
            self.details[name] = output
        return status

    @property
    def failed(self) -> int:
        return sum(1 for status, _, _ in self.rows if status == FAIL)

    def summarise(self) -> int:
        counts = {PASS: 0, FAIL: 0, SKIP: 0}
        for status, _, _ in self.rows:
            counts[status] += 1
        print()
        for name, output in self.details.items():
            print(f"--- {name} ---")
            for line in output.strip().splitlines()[-30:]:
                print(f"  | {line}")
            print()
        print(f"{counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped")
        return 1 if counts[FAIL] else 0


def run(command: list[str], cwd: Path | None = None, env: dict | None = None,
        timeout: float = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=str(cwd) if cwd else None, env=env, timeout=timeout,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def tail(process: subprocess.CompletedProcess) -> str:
    return ((process.stdout or "") + "\n" + (process.stderr or "")).strip()


# ------------------------------------------------------------------ discovery


def find_python311() -> str | None:
    """An interpreter to build clean environments with.

    3.11 because that is what the image runs and what ``amsc-poc`` declares
    support for; the interpreter running this script is deliberately not
    reused, since it is the developer's and has everything already.
    """
    if sys.platform == "win32":
        found = run(["py", "-3.11", "-c", "import sys; print(sys.executable)"])
        if found.returncode == 0 and found.stdout.strip():
            return found.stdout.strip()
    for candidate in ("python3.11", "python3", "python"):
        where = shutil.which(candidate)
        if not where:
            continue
        version = run([where, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
        if version.returncode == 0 and version.stdout.strip() == "3.11":
            return where
    return None


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def docker_version() -> str | None:
    if not shutil.which("docker"):
        return None
    found = run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=60)
    return found.stdout.strip() if found.returncode == 0 and found.stdout.strip() else None


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def fingerprint() -> dict:
    found = {}
    for relative in DEVELOPER_STATE:
        path = ROOT / relative
        try:
            found[relative] = (path.stat().st_size,
                               hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            found[relative] = None
    return found


# ------------------------------------------------------- reading the manifest

PIN_LINE = re.compile(r"^amsc-poc\s*@\s*(git\+\S+?)@([0-9a-f]{40})\s*$", re.M)


def read_pin(requirements: Path) -> tuple[str, str] | None:
    match = PIN_LINE.search(requirements.read_text(encoding="utf-8"))
    return (match.group(1), match.group(2)) if match else None


def isolated_env(data_dir: Path, extra: dict | None = None) -> dict:
    """An environment in which nothing of this machine can answer.

    The state root is a throwaway directory; the model caches are pointed at
    another one and then switched offline, so an import that wanted weights
    fails loudly here instead of quietly succeeding from the developer's cache
    and failing on a clean machine.
    """
    env = dict(os.environ)
    env["CHAT_RAG_DATA_DIR"] = str(data_dir)
    env.pop("VECTOR_DB_PATH", None)
    env.pop("STRUCTURED_PARSER_CACHE", None)
    env["RETRIEVAL_PROFILE"] = "bm25_only"
    env["HF_HOME"] = str(data_dir / "hf")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # Not set on purpose: an entrypoint that needs it to survive its own
    # start-up banner is one that needs a particular launcher, which is the
    # sort of hidden dependency this gate exists to find.
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


# =========================================================== the checks


def check_clone(report: Report, work: Path, local: bool = False) -> Path | None:
    """A clone, and proof of what it is a clone of.

    Two separate claims, reported separately because they fail for different
    reasons and one does not invalidate the other:

    ``clone``       the branch is pushed and the remote's tip is this checkout,
                    so the gate is testing the work being reported on;
    ``clone.clean`` the tracked tree carries no developer state -- no ``.env``,
                    no knowledge base records, no vector store. A clone that
                    arrived with one would let a later check pass because of
                    this machine rather than because of the source.
    """
    head = run(["git", "rev-parse", "HEAD"], cwd=ROOT).stdout.strip()
    target = work / "chat_rag"

    if local:
        # Committed state only -- `git clone` of a path never copies the
        # working tree -- so this still proves the tracked source is
        # sufficient. It does not prove the branch is pushed, which is why it
        # has to be asked for.
        cloned = run(["git", "clone", "--no-hardlinks", str(ROOT), str(target)], timeout=1800)
        if cloned.returncode != 0:
            report.add(FAIL, "clone", "git clone of this checkout failed", tail(cloned))
            return None
        report.add(SKIP, "clone",
                   f"--local: cloned this checkout at {head[:7]}, not origin; "
                   "pushed-ness is NOT proved")
    else:
        remote = run(["git", "remote", "get-url", "origin"], cwd=ROOT).stdout.strip()
        if not remote:
            report.add(SKIP, "clone", "no origin remote to clone from")
            return None
        listed = run(["git", "ls-remote", "--heads", remote], timeout=180)
        if listed.returncode != 0:
            report.add(SKIP, "clone", f"{remote} unreachable (offline?)", tail(listed))
            return None
        # The question is whether *this commit* is on the remote, not whether a
        # branch of a particular name is. Asked this way it also answers on a
        # detached checkout, which is what a CI runner hands you.
        published = [line.split()[1] for line in listed.stdout.splitlines()
                     if line.split() and line.split()[0] == head]
        if not published:
            report.add(
                FAIL, "clone",
                f"{head[:7]} is on no branch of origin: push first, or the gate proves "
                "something other than your work (--local clones this checkout instead)",
            )
            return None
        ref = published[0]
        # refs/heads/refactor/productionization -> refactor/productionization.
        # Splitting on the last slash loses everything before it, and branch
        # names here have slashes in them.
        name = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
        cloned = run(["git", "clone", "--depth", "1", "--branch", name, remote, str(target)],
                     timeout=1800)
        if cloned.returncode != 0:
            report.add(FAIL, "clone", f"git clone of {name} failed", tail(cloned))
            return None
        report.add(PASS, "clone", f"origin/{name} @ {head[:7]}, the commit this checkout is on")

    leaked = [name for name in MUST_NOT_BE_CLONED if (target / name).exists()]
    if leaked:
        report.add(FAIL, "clone.clean",
                   "the clone carries developer state, so a later check could pass "
                   "because of this machine: " + ", ".join(leaked))
    else:
        report.add(PASS, "clone.clean",
                   f"none of the {len(MUST_NOT_BE_CLONED)} developer state paths are tracked")
    return target


def check_pin_shape(report: Report, clone: Path) -> tuple[str, str] | None:
    """The dependency line pip has to read, read the way pip reads it."""
    pin = read_pin(clone / "requirements.txt")
    if pin is None:
        report.add(FAIL, "pin.shape",
                   "requirements.txt does not pin amsc-poc to a 40-character commit "
                   "on a line of its own")
        return None
    url, commit = pin
    report.add(PASS, "pin.shape", f"amsc-poc @ {commit[:7]}, on a line of its own")
    return url, commit


def check_chunk(report: Report, work: Path, python311: str | None,
                pin: tuple[str, str] | None) -> None:
    """The pinned dependency: fetchable, installable, importable, and it builds
    the product's own Viewer shell.

    Installed the way ``requirements.txt`` asks for it rather than cloned and
    built by hand -- that is the artifact chat_rag actually consumes, and a
    clone would prove something adjacent to it.
    """
    if pin is None:
        report.add(SKIP, "chunk.install", "no usable pin to install")
        return
    if python311 is None:
        report.add(SKIP, "chunk.install", "no Python 3.11 on this machine")
        return
    url, commit = pin

    venv = work / "chunk-venv"
    made = run([python311, "-m", "venv", str(venv)], timeout=600)
    if made.returncode != 0:
        report.add(FAIL, "chunk.install", "could not create a clean 3.11 venv", tail(made))
        return
    python = venv_python(venv)

    installed = run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                     "--no-cache-dir", f"amsc-poc @ {url}@{commit}"], timeout=1800)
    if installed.returncode != 0:
        report.add(FAIL, "chunk.install",
                   f"pip could not install the pinned revision {commit[:7]}", tail(installed))
        return
    report.add(PASS, "chunk.install", f"pip installed {commit[:7]} into a fresh 3.11 venv")

    probe = (
        "import json, amsc, importlib.metadata as md;"
        "d = md.distribution('amsc-poc');"
        "raw = d.read_text('direct_url.json') or '{}';"
        "info = json.loads(raw);"
        "print(json.dumps({"
        "'file': amsc.__file__,"
        "'commit': (info.get('vcs_info') or {}).get('commit_id'),"
        "'editable': bool((info.get('dir_info') or {}).get('editable')),"
        "}))"
    )
    imported = run([str(python), "-c", probe], cwd=work, timeout=600)
    if imported.returncode != 0:
        report.add(FAIL, "chunk.import", "the installed package does not import", tail(imported))
        return
    facts = json.loads(imported.stdout.strip().splitlines()[-1])
    problems = []
    if facts["editable"]:
        problems.append("installed as an editable checkout")
    if facts["commit"] != commit:
        problems.append(f"reports commit {facts['commit']}, not {commit}")
    if str(ROOT.parent / "chunk").lower() in (facts["file"] or "").lower():
        problems.append(f"resolved to the developer's checkout at {facts['file']}")
    if problems:
        report.add(FAIL, "chunk.import", "; ".join(problems))
        return
    report.add(PASS, "chunk.import",
               f"amsc {commit[:7]} from site-packages, not an editable sibling")

    # --- the product Viewer shell, built from the installed package alone ---
    output = work / "viewer-v3" / "index.html"
    again = work / "viewer-v3-again" / "index.html"
    env = isolated_env(work / "viewer-state")
    builds = [run([str(python), "-m", "amsc.viewer.build", "--output", str(path)],
                  cwd=work, env=env, timeout=600) for path in (output, again)]
    if any(build.returncode != 0 for build in builds):
        report.add(FAIL, "chunk.viewer", "the Viewer v3 shell build failed",
                   tail(builds[0] if builds[0].returncode else builds[1]))
        return
    try:
        result = json.loads(builds[0].stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        report.add(FAIL, "chunk.viewer", "the build printed no result", tail(builds[0]))
        return
    if result.get("embedded_documents") != 0:
        report.add(FAIL, "chunk.viewer",
                   f"the shell embedded {result.get('embedded_documents')} documents; "
                   "a fresh clone has no corpus to embed")
        return
    size = output.stat().st_size
    if size < 50_000:
        report.add(FAIL, "chunk.viewer", f"the page is only {size} bytes; that is not a build")
        return
    if output.read_bytes() != again.read_bytes():
        report.add(FAIL, "chunk.viewer", "two builds of the same shell differ")
        return
    report.add(PASS, "chunk.viewer",
               f"shell built twice, byte-identical, {size // 1024} KiB, 0 documents embedded")


def check_host_install(report: Report, work: Path, clone: Path | None,
                       python311: str | None, requested: bool) -> Path | None:
    """The same install, on this machine, outside a container."""
    if not requested:
        report.add(SKIP, "install.host",
                   "not run -- pass --with-host-install (several minutes, ~1 GB of wheels)")
        return None
    if clone is None or python311 is None:
        report.add(SKIP, "install.host", "needs a clone and a Python 3.11")
        return None

    venv = work / "chat_rag-venv"
    made = run([python311, "-m", "venv", str(venv)], timeout=900)
    if made.returncode != 0:
        report.add(FAIL, "install.host", "could not create a clean 3.11 venv", tail(made))
        return None
    python = venv_python(venv)

    started = time.monotonic()
    installed = run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                     "-r", "requirements.txt"], cwd=clone, timeout=5400)
    if installed.returncode != 0:
        report.add(FAIL, "install.host", "pip install -r requirements.txt failed", tail(installed))
        return None
    minutes = (time.monotonic() - started) / 60

    origin = run([str(python), "-c",
                  "import json, importlib.metadata as md;"
                  "i = json.loads(md.distribution('amsc-poc').read_text('direct_url.json') or '{}');"
                  "print((i.get('vcs_info') or {}).get('commit_id'),"
                  "      bool((i.get('dir_info') or {}).get('editable')))"], timeout=300)
    report.add(PASS, "install.host",
               f"requirements.txt installed into a fresh 3.11 venv in {minutes:.1f} min "
               f"(amsc {origin.stdout.split()[0][:7] if origin.stdout.split() else '?'}, "
               f"editable={origin.stdout.split()[1] if len(origin.stdout.split()) > 1 else '?'})")
    return python


def check_smokes(report: Report, work: Path, clone: Path | None, python: Path | None,
                 host_requested: bool) -> None:
    """The two smokes this repository already ships, run against the clone."""
    if python is None or clone is None:
        reason = ("covered by the image build; pass --with-host-install to run it here too"
                  if not host_requested else "no host install to run them against")
        report.add(SKIP, "import.smoke", reason)
        report.add(SKIP, "serve.smoke", reason)
        return

    env = isolated_env(work / "smoke-state")
    smoke = run([str(python), "tools/import_smoke.py"], cwd=clone, env=env, timeout=1800)
    report.add(PASS if smoke.returncode == 0 else FAIL, "import.smoke",
               "every product module imports from the declared dependencies alone"
               if smoke.returncode == 0 else "import_smoke.py failed",
               tail(smoke))

    serve = run([str(python), "tools/serve_smoke.py"], cwd=clone, env=env, timeout=1800)
    report.add(PASS if serve.returncode == 0 else FAIL, "serve.smoke",
               "python -m wsgi binds, answers /api/health on waitress, and stops"
               if serve.returncode == 0 else "serve_smoke.py failed",
               tail(serve))


def check_docker(report: Report, clone: Path | None, docker: str | None,
                 no_cache: bool = False, asked_to_skip: bool = False) -> None:
    """Build and run the tracked repository as a container.

    This is the strongest clean-environment proof available: the base image has
    no editable siblings, no developer packages and no caches, and the build
    already runs the import and serve smokes inside itself, so a build that
    succeeds has proved rather more than that pip resolved.
    """
    if docker is None:
        # A SKIP has to say which kind it is: a capability this machine does
        # not have, or one the caller turned off.
        why = "skipped by --no-docker" if asked_to_skip else "no Docker daemon on this machine"
        report.add(SKIP, "docker.build", why)
        report.add(SKIP, "docker.run", why)
        return
    if clone is None:
        report.add(SKIP, "docker.build", "no clone to build from")
        report.add(SKIP, "docker.run", "no clone to build from")
        return

    tag = "chat-rag:repro-gate"
    command = ["docker", "build", "-t", tag, "."]
    if no_cache:
        command.insert(2, "--no-cache")
    started = time.monotonic()
    built = run(command, cwd=clone, timeout=5400)
    if built.returncode != 0:
        report.add(FAIL, "docker.build", "docker build failed", tail(built))
        return
    minutes = (time.monotonic() - started) / 60
    # Said out loud, because it changes what the check proves. A build that
    # reused cached layers did not re-resolve the dependencies, so it cannot
    # speak to whether they still install; a cold one did. CI runners are
    # always cold, which is where that proof comes from by default.
    reused = "CACHED" in (built.stdout or "") + (built.stderr or "")
    how = ("no cache, every layer rebuilt" if no_cache
           else "reused cached layers -- CI builds cold; --docker-no-cache does here"
           if reused else "no cache was available, every layer rebuilt")
    report.add(PASS, "docker.build",
               f"built from the clone in {minutes:.1f} min ({how}); "
               "the import and serve smokes ran inside it")

    name = "chat-rag-repro-gate"
    port = free_port()
    run(["docker", "rm", "-f", name], timeout=120)
    up = run(["docker", "run", "-d", "--name", name, "-p", f"{port}:5005",
              "-e", "RETRIEVAL_PROFILE=bm25_only", tag], timeout=300)
    if up.returncode != 0:
        report.add(FAIL, "docker.run", "docker run failed", tail(up))
        return
    try:
        health = None
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/health", timeout=5
                ) as response:
                    health = (response.status, response.headers.get("Server", ""),
                              json.loads(response.read().decode("utf-8") or "{}"))
                    break
            except (urllib.error.URLError, OSError, ValueError):
                time.sleep(1)
        if health is None:
            logs = run(["docker", "logs", "--tail", "40", name], timeout=120)
            report.add(FAIL, "docker.run", "the container never answered /api/health", tail(logs))
            return
        status, server, body = health
        if status != 200 or body.get("status") != "healthy":
            report.add(FAIL, "docker.run", f"/api/health answered {status} {body!r}")
            return
        if "waitress" not in server.lower():
            report.add(FAIL, "docker.run", f"served by {server!r}, not the production server")
            return
        report.add(PASS, "docker.run", f"health 200 on {server}, in a container built from the clone")

        # Where the running container put its state. /data is the mount point;
        # /app is the image, and a write there would be state in a layer.
        listing = run(["docker", "exec", name, "sh", "-c",
                       "ls -1 /data; echo ---; ls -a /app | "
                       "grep -E 'chroma_db|knowledge_bases|ingested_documents|^[.]cache' "
                       "|| true"], timeout=120)
        under_data, _, in_app = listing.stdout.partition("---")
        if in_app.strip():
            report.add(FAIL, "docker.state",
                       "the container wrote state into the image: " + in_app.split()[0])
        else:
            report.add(PASS, "docker.state",
                       "state under /data only (" + ", ".join(under_data.split()) + "), none in /app")
    finally:
        run(["docker", "rm", "-f", name], timeout=180)
        run(["docker", "rmi", tag], timeout=300)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-host-install", action="store_true",
                        help="also install requirements.txt into a fresh venv on this machine")
    parser.add_argument("--no-docker", action="store_true", help="skip the container checks")
    parser.add_argument("--docker-no-cache", action="store_true",
                        help="build the image without the layer cache, so the dependency "
                             "install is really re-run (slow; CI is cold anyway)")
    parser.add_argument("--local", action="store_true",
                        help="clone this checkout instead of origin, to run the gate before "
                             "pushing; the pushed-ness check then reports SKIP")
    parser.add_argument("--keep", action="store_true",
                        help="leave the temporary clone and environments in place")
    args = parser.parse_args(argv)

    python311 = find_python311()
    docker = None if args.no_docker else docker_version()

    print("chat_rag reproducibility gate")
    print(f"  driver   python {platform.python_version()} on {sys.platform}")
    print(f"  clean    python {'3.11 at ' + python311 if python311 else '3.11 NOT FOUND'}")
    print(f"  docker   {docker or ('skipped' if args.no_docker else 'not available')}")
    print()

    before = fingerprint()
    work = Path(tempfile.mkdtemp(prefix="chat_rag-repro-"))
    report = Report()
    try:
        clone = check_clone(report, work, local=args.local)
        pin = check_pin_shape(report, clone) if clone else None
        check_chunk(report, work, python311, pin)
        python = check_host_install(report, work, clone, python311, args.with_host_install)
        check_smokes(report, work, clone, python, args.with_host_install)
        check_docker(report, clone, docker, no_cache=args.docker_no_cache,
                     asked_to_skip=args.no_docker)

        changed = [name for name, was in before.items() if fingerprint().get(name) != was]
        report.add(FAIL if changed else PASS, "state.untouched",
                   "the developer's knowledge bases, ledger, gold set and Chroma store "
                   "are unchanged" if not changed else "changed: " + ", ".join(changed))
    finally:
        if args.keep:
            print(f"\n  kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    return report.summarise()


if __name__ == "__main__":
    raise SystemExit(main())
