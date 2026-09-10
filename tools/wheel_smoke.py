"""Build the wheel, install it into an empty environment, and use it.

    python tools/wheel_smoke.py
    python tools/wheel_smoke.py --with-extras     # also prove [all] adds them
    python tools/wheel_smoke.py --keep

``tools/import_smoke.py`` proves the *product* imports from its declared
dependencies -- but it runs in a checkout, against a repository that already
has ``interfaces/``, ``asgi.py`` and a ``.env`` beside it. This proves the
next claim along, which is the one a consumer depends on: that
``pip install chat-rag`` in an environment with nothing else in it produces a
library that imports, exposes the surface it promises, and refuses what it
cannot do by name rather than by ImportError.

What it checks
--------------

* the wheel **builds** from this checkout;
* it **installs** into a fresh interpreter with no wheels warmed and no
  sibling checkout to fall back on;
* ``import chat_rag`` costs nothing -- no torch, no sentence-transformers, no
  PDF stack in ``sys.modules`` afterwards, which is what makes the extras real
  rather than a metadata gesture;
* the **published surface** is exactly ``chat_rag.api.__all__``, and every
  name on it resolves;
* what the extras leave out is **refused by name**: asking for a local
  embedding model without ``[local]`` says which extra to install, rather than
  failing with a traceback from three frames down;
* the adapter is **not there**: ``import interfaces`` and ``import asgi`` fail,
  because they are not the library and never shipped.

``--with-extras`` then installs ``chat-rag[all]`` into a second environment and
checks the other half of the same claim: the local model can now be *built*,
and it is building it -- not importing the library -- that loads torch. An
extra changes what the engine can do, never what importing it costs. It is off
by default because it downloads torch and a set of model weights.

The one thing this cannot check
-------------------------------

``amsc-poc`` is not on any index. It is passed to pip here as the pinned
``git+https`` requirement ``requirements.txt`` names, which is exactly what a
consumer would have to do today -- and is the reason this library is not yet
``pip install chat-rag`` for somebody outside this repository.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^(amsc-poc\s*@\s*git\+\S+)$", re.M)


def pinned_amsc() -> str | None:
    """The ``amsc-poc`` requirement, read from the file that owns the pin."""
    found = PIN.search((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    return found.group(1).strip() if found else None


def run(command: list[str], cwd: Path | None = None,
        timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=str(cwd) if cwd else None, timeout=timeout,
                          capture_output=True, text=True)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python")


def tail(finished: subprocess.CompletedProcess, lines: int = 25) -> str:
    output = (finished.stdout or "") + "\n" + (finished.stderr or "")
    return "\n".join(f"  | {line}" for line in output.strip().splitlines()[-lines:])


# --------------------------------------------------------------- the probe
#
# Run inside the clean interpreter. It prints one JSON line, so this side
# reports rather than parses prose.
PROBE = r"""
import json, sys

report = {}

import chat_rag
report["version"] = chat_rag.__version__
report["published"] = sorted(chat_rag.__all__)

from chat_rag import Engine, EngineConfig, Settings
import chat_rag.api as api
report["api_published"] = sorted(api.__all__)
report["resolves"] = sorted(n for n in api.__all__ if getattr(api, n, None) is not None)

HEAVY = ("torch", "sentence_transformers", "ollama", "fitz", "pymupdf",
         "pymupdf4llm", "docx")


def loaded():
    return sorted(m for m in HEAVY if m in sys.modules)


# Importing the surface must not drag in what the extras exist to leave out --
# and that is true whether or not they are installed, because it is the import
# that is lazy, not the package that is absent.
report["heavy_after_import"] = loaded()

# The configuration is a value: constructing one reads nothing and needs no
# database, so this works in an environment that has never seen PostgreSQL.
settings = EngineConfig(retrieval_profile="bm25_only", top_k=3).build()
report["profile"] = settings.retrieval_profile
report["top_k"] = settings.default_top_k
report["settings_is_public"] = Settings is type(settings)

# The migrations have to be *in* the installed package: a library that cannot
# create its own schema is not installable in any useful sense.
from pathlib import Path
import chat_rag.storage as storage
versions = Path(storage.__file__).parent / "migrations" / "versions"
report["migrations"] = sorted(p.name for p in versions.glob("*.py"))

# What the extras decide is what the engine can *do*, not what importing it
# costs. Without [local] this is refused by name; with it, the model is built
# -- and only then does the heavy stack appear.
from chat_rag.core.exceptions import ConfigurationException
try:
    from chat_rag.components.embedding import SentenceTransformerEmbedding
    SentenceTransformerEmbedding("all-MiniLM-L6-v2")
except ConfigurationException as refused:
    report["local_refusal"] = str(refused)
except ImportError as raw:
    report["local_refusal"] = "ImportError: " + str(raw)
else:
    report["local_refusal"] = None
report["heavy_after_local"] = loaded()

# And the adapter is not here, because it is not the library.
for name in ("interfaces", "asgi", "cli"):
    try:
        __import__(name)
        report.setdefault("leaked", []).append(name)
    except ImportError:
        pass

print("@@" + json.dumps(report))
"""


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, name: str, detail: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<24} {detail}", flush=True)
        if not ok:
            self.failures.append(name)

    def finish(self) -> int:
        print()
        if self.failures:
            print(f"{len(self.failures)} check(s) failed: {', '.join(self.failures)}")
            return 1
        print("the wheel installs into an empty environment and is usable there")
        return 0


def build_wheel(report: Report, out: Path) -> Path | None:
    built = run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                 "--outdir", str(out), str(ROOT)])
    if built.returncode != 0:
        report.check(False, "wheel.build", "python -m build failed")
        print(tail(built))
        return None
    wheels = sorted(out.glob("*.whl"))
    if len(wheels) != 1:
        report.check(False, "wheel.build", f"expected one wheel, got {len(wheels)}")
        return None
    report.check(True, "wheel.build", wheels[0].name)
    return wheels[0]


def install(report: Report, name: str, venv: Path, requirement: str,
            amsc: str | None) -> Path | None:
    """A fresh interpreter with exactly ``requirement`` and its dependencies."""
    made = run([sys.executable, "-m", "venv", str(venv)], timeout=900)
    if made.returncode != 0:
        report.check(False, name, "could not create a clean venv")
        print(tail(made))
        return None
    python = venv_python(venv)

    # amsc-poc is not on an index; it is named here exactly as requirements.txt
    # pins it, which is what a consumer has to do today.
    wanted = [requirement] + ([amsc] if amsc else [])
    installed = run([str(python), "-m", "pip", "install",
                     "--disable-pip-version-check", *wanted], timeout=5400)
    if installed.returncode != 0:
        report.check(False, name, "pip install failed")
        print(tail(installed))
        return None
    report.check(True, name, f"installed {requirement.split('/')[-1]} into a clean venv")
    return python


def probe(report: Report, python: Path, *, expect_heavy: bool) -> dict | None:
    """Use the installed library, in its own interpreter, from a directory
    that is not this checkout -- so nothing can pass by being on the path."""
    where = python.parent.parent
    finished = run([str(python), "-c", PROBE], cwd=where, timeout=1800)
    if finished.returncode != 0:
        report.check(False, "wheel.import", "the installed library could not be used")
        print(tail(finished, 40))
        return None
    line = next((l for l in finished.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        report.check(False, "wheel.import", "the probe printed no report")
        print(tail(finished, 40))
        return None
    found = json.loads(line[2:])

    report.check(True, "wheel.import", f"chat_rag {found['version']} imports and answers")
    report.check(found["heavy_after_import"] == [], "import.cost",
                 "importing the surface loads no torch and no PDF stack"
                 if not found["heavy_after_import"]
                 else f"loaded {found['heavy_after_import']}")
    report.check(found["published"] == found["api_published"],
                 "wheel.surface",
                 f"{len(found['published'])} published names, both import paths agreeing")
    report.check(found["resolves"] == found["api_published"],
                 "wheel.resolves", "every published name resolves")
    report.check(found["settings_is_public"] is True,
                 "wheel.settings", "EngineConfig.build() returns the published Settings")
    report.check(bool(found["migrations"]),
                 "wheel.migrations", f"the schema ships: {', '.join(found['migrations'])}")
    report.check(not found.get("leaked"),
                 "wheel.boundary",
                 "interfaces, asgi and cli are not installed"
                 if not found.get("leaked") else f"leaked: {found['leaked']}")

    if expect_heavy:
        report.check(found["local_refusal"] is None, "extras.local",
                     "a local embedding model can be built")
        # And it is *this* that pulls the stack in -- which is the whole point
        # of the extra: it changes what the engine can do, not what importing
        # it costs.
        report.check("sentence_transformers" in found["heavy_after_local"],
                     "extras.deferred",
                     "the local stack loads when a model is built, not before: "
                     + ", ".join(found["heavy_after_local"]))
    else:
        refusal = found["local_refusal"] or ""
        report.check("chat-rag[local]" in refusal, "extras.refusal",
                     "asking for a local model names the extra to install"
                     if "chat-rag[local]" in refusal else f"said: {refusal[:120]!r}")
        report.check(found["heavy_after_local"] == [], "extras.absent",
                     "and nothing heavy was loaded on the way to that refusal"
                     if not found["heavy_after_local"]
                     else f"loaded {found['heavy_after_local']}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-extras", action="store_true",
                        help="also install chat-rag[all] and check the extras arrive "
                             "(downloads torch)")
    parser.add_argument("--keep", action="store_true",
                        help="leave the wheel and the environments in place")
    args = parser.parse_args(argv)

    print("chat_rag wheel smoke")
    print(f"  driver   python {platform.python_version()} on {sys.platform}")
    amsc = pinned_amsc()
    print(f"  amsc     {'pinned in requirements.txt' if amsc else 'NOT FOUND in requirements.txt'}")
    print()

    work = Path(tempfile.mkdtemp(prefix="chat_rag-wheel-"))
    report = Report()
    try:
        wheel = build_wheel(report, work / "dist")
        if wheel is None:
            return report.finish()

        python = install(report, "wheel.install", work / "bare", str(wheel), amsc)
        if python is not None:
            probe(report, python, expect_heavy=False)

        if args.with_extras:
            extras = install(report, "extras.install", work / "full",
                             f"{wheel}[all]", amsc)
            if extras is not None:
                probe(report, extras, expect_heavy=True)
        else:
            print("  SKIP  extras.install          "
                  "not run -- pass --with-extras (downloads torch)")
    finally:
        if args.keep:
            print(f"\n  kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
