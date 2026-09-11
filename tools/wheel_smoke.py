"""Build the distribution, install it into an empty environment, and use it.

    python tools/wheel_smoke.py
    python tools/wheel_smoke.py --sdist                # the source distribution too
    python tools/wheel_smoke.py --database-url URL     # and the first use, for real
    python tools/wheel_smoke.py --with-extras          # also prove [all] adds them
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

* the wheel **builds** from this checkout (and, with ``--sdist``, so does the
  source distribution, which is then installed the same way -- a build from
  source in an isolated environment, which is what a consumer without a wheel
  gets);
* it **installs** into a fresh interpreter with no wheels warmed and no
  sibling checkout to fall back on -- and with nothing named but the
  artifact. ``amsc-poc`` is on no index; the wheel's own metadata says where
  it comes from (``pyproject.toml``), and this is where that claim is tested
  rather than worked around;
* ``import chat_rag`` costs nothing -- no torch, no sentence-transformers, no
  PDF stack in ``sys.modules`` afterwards, which is what makes the extras real
  rather than a metadata gesture;
* the **published surface** is exactly ``chat_rag.api.__all__``, and every
  name on it resolves;
* what the extras leave out is **refused by name**: asking for a local
  embedding model without ``[local]`` says which extra to install, rather than
  failing with a traceback from three frames down;
* the adapter is **not there**: ``import interfaces`` and ``import asgi`` fail,
  because they are not the library and never shipped;
* given ``--database-url``, the **first use**: ``migrate_database`` builds
  the schema in an empty database and reports ``current`` the second time;
  an ``Engine`` on it creates a knowledge base, ingests a Markdown document
  with no extra installed and no provider configured, and searches it
  lexically. The URL should name an empty database of its own -- the schema
  is created in it and nothing is dropped afterwards.

``--with-extras`` then installs ``chat-rag[all]`` into a second environment and
checks the other half of the same claim: the local model can now be *built*,
and it is building it -- not importing the library -- that loads torch. An
extra changes what the engine can do, never what importing it costs. It is off
by default because it downloads torch and a set of model weights.

What it needs
-------------

A network -- ``amsc-poc`` is fetched from its pinned commit, the rest from
the index -- and ``git`` on the installing machine, for the same reason the
Docker build installs it. ``tests/integration/test_clean_install.py`` drives
the same functions from the suite, against a throwaway database.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path | None = None,
        timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=str(cwd) if cwd else None, timeout=timeout,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python")


def tail(finished: subprocess.CompletedProcess, lines: int = 25) -> str:
    output = (finished.stdout or "") + "\n" + (finished.stderr or "")
    return "\n".join(f"  | {line}" for line in output.strip().splitlines()[-lines:])


# --------------------------------------------------------------- the probe
#
# Run inside the clean interpreter. It prints one JSON line, so this side
# reports rather than parses prose. ``sys.argv[1]``, when given, is a database
# URL: the first-use half runs only then.
PROBE = r"""
import json, os, sys, tempfile

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

# The first use, when a database was given: the schema from the package's own
# migrations, then a knowledge base, a document and a search on it -- with no
# extra installed and no provider configured, which is the smallest engine
# there is.
url = sys.argv[1] if len(sys.argv) > 1 else ""
if url:
    from chat_rag import migrate_database

    first = migrate_database(database_url=url)
    second = migrate_database(database_url=url)
    report["migration"] = [first.outcome, second.outcome]
    report["schema_head"] = first.after

    # Under the interpreter's own directory, which the caller removes; a
    # system temp directory would outlive the run.
    data = tempfile.mkdtemp(prefix="smoke-data-", dir=os.getcwd())
    document = os.path.join(data, "notes.md")
    with open(document, "w", encoding="utf-8") as handle:
        handle.write(
            "# Liquidity\n\nThe liquidity coverage ratio stayed above the "
            "regulatory minimum through the year.\n\n# Credit\n\nNon-performing "
            "loans were provisioned in full.\n"
        )
    config = EngineConfig(database_url=url, data_dir=data,
                          retrieval_profile="bm25_only", read_environment=False)
    with Engine(config) as engine:
        kb = engine.knowledge_bases.create("Smoke")
        ingested = kb.ingest(document)
        hits = kb.search("liquidity coverage", method="bm25", limit=3)
        report["first_use"] = {
            "knowledge_bases": [k.name for k in engine.knowledge_bases.list()],
            "chunks": ingested.chunk_count,
            "hits": len(hits),
            "top_hit_mentions_liquidity": bool(hits) and "liquidity" in hits[0].content.lower(),
            "health": engine.health().state,
        }
    report["heavy_after_first_use"] = loaded()

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
        print("the distribution installs into an empty environment and is usable there")
        return 0


def build(report: Report, out: Path, *, sdist: bool = False) -> dict[str, Path]:
    """The artifacts, built from this checkout the way a release would be.

    ``--no-isolation`` because the build backend is already installed here
    and a fresh isolated environment would download setuptools to prove a
    claim about this repository's packaging. Returns ``{"wheel": path}`` and,
    when asked, ``"sdist"`` too; empty when the build failed.
    """
    command = [sys.executable, "-m", "build", "--wheel", "--no-isolation",
               "--outdir", str(out), str(ROOT)]
    if sdist:
        command.insert(4, "--sdist")
    built = run(command)
    if built.returncode != 0:
        report.check(False, "dist.build", "python -m build failed")
        print(tail(built))
        return {}
    wheels = sorted(out.glob("*.whl"))
    sdists = sorted(out.glob("*.tar.gz"))
    if len(wheels) != 1 or (sdist and len(sdists) != 1):
        report.check(False, "dist.build",
                     f"expected one wheel{' and one sdist' if sdist else ''}, got "
                     f"{[p.name for p in wheels + sdists]}")
        return {}
    artifacts = {"wheel": wheels[0]}
    if sdist:
        artifacts["sdist"] = sdists[0]
    report.check(True, "dist.build", ", ".join(p.name for p in artifacts.values()))
    return artifacts


def install(report: Report, name: str, venv: Path, requirement: str) -> Path | None:
    """A fresh interpreter with exactly ``requirement`` and its dependencies.

    Nothing else is named. ``amsc-poc`` in particular is not: the artifact's
    own metadata says where it comes from, and a consumer's pip has to be
    able to follow that on its own.
    """
    made = run([sys.executable, "-m", "venv", str(venv)], timeout=900)
    if made.returncode != 0:
        report.check(False, name, "could not create a clean venv")
        print(tail(made))
        return None
    python = venv_python(venv)

    installed = run([str(python), "-m", "pip", "install",
                     "--disable-pip-version-check", requirement], timeout=5400)
    if installed.returncode != 0:
        report.check(False, name, "pip install failed")
        print(tail(installed))
        return None
    report.check(True, name, f"installed {Path(requirement.split('[')[0]).name} "
                             "into a clean venv, naming nothing else")
    return python


def probe(report: Report, python: Path, *, expect_heavy: bool,
          database_url: str | None = None, prefix: str = "wheel") -> dict | None:
    """Use the installed library, in its own interpreter, from a directory
    that is not this checkout -- so nothing can pass by being on the path."""
    where = python.parent.parent
    command = [str(python), "-c", PROBE] + ([database_url] if database_url else [])
    finished = run(command, cwd=where, timeout=1800)
    if finished.returncode != 0:
        report.check(False, f"{prefix}.import", "the installed library could not be used")
        print(tail(finished, 40))
        return None
    line = next((l for l in finished.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        report.check(False, f"{prefix}.import", "the probe printed no report")
        print(tail(finished, 40))
        return None
    found = json.loads(line[2:])

    report.check(True, f"{prefix}.import", f"chat_rag {found['version']} imports and answers")
    report.check(found["heavy_after_import"] == [], f"{prefix}.import.cost",
                 "importing the surface loads no torch and no PDF stack"
                 if not found["heavy_after_import"]
                 else f"loaded {found['heavy_after_import']}")
    report.check(found["published"] == found["api_published"],
                 f"{prefix}.surface",
                 f"{len(found['published'])} published names, both import paths agreeing")
    report.check(found["resolves"] == found["api_published"],
                 f"{prefix}.resolves", "every published name resolves")
    report.check(found["settings_is_public"] is True,
                 f"{prefix}.settings", "EngineConfig.build() returns the published Settings")
    report.check(bool(found["migrations"]),
                 f"{prefix}.migrations", f"the schema ships: {', '.join(found['migrations'])}")
    report.check(not found.get("leaked"),
                 f"{prefix}.boundary",
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
        report.check("chat-rag[local]" in refusal, f"{prefix}.extras.refusal",
                     "asking for a local model names the extra to install"
                     if "chat-rag[local]" in refusal else f"said: {refusal[:120]!r}")
        report.check(found["heavy_after_local"] == [], f"{prefix}.extras.absent",
                     "and nothing heavy was loaded on the way to that refusal"
                     if not found["heavy_after_local"]
                     else f"loaded {found['heavy_after_local']}")

    if database_url:
        migration = found.get("migration") or []
        report.check(migration == ["created", "current"], f"{prefix}.migrate",
                     f"migrate_database() created the schema at {found.get('schema_head')} "
                     "and reported current the second time"
                     if migration == ["created", "current"] else f"reported {migration}")
        use = found.get("first_use") or {}
        worked = (use.get("knowledge_bases") == ["Smoke"] and use.get("chunks", 0) > 0
                  and use.get("hits", 0) > 0 and use.get("top_hit_mentions_liquidity"))
        report.check(worked, f"{prefix}.first_use",
                     f"a knowledge base, {use.get('chunks')} chunks from a Markdown "
                     f"document, {use.get('hits')} lexical hits, health {use.get('health')!r}"
                     if worked else f"first use fell short: {use}")
        report.check(found.get("heavy_after_first_use") == [], f"{prefix}.first_use.cost",
                     "and none of it loaded torch or the PDF stack"
                     if not found.get("heavy_after_first_use")
                     else f"loaded {found.get('heavy_after_first_use')}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sdist", action="store_true",
                        help="also build the source distribution and install from it")
    parser.add_argument("--database-url", default=None, metavar="URL",
                        help="an empty PostgreSQL database to migrate and use from the "
                             "installed library (the schema is created; nothing is dropped)")
    parser.add_argument("--with-extras", action="store_true",
                        help="also install chat-rag[all] and check the extras arrive "
                             "(downloads torch)")
    parser.add_argument("--keep", action="store_true",
                        help="leave the artifacts and the environments in place")
    args = parser.parse_args(argv)

    print("chat_rag distribution smoke")
    print(f"  driver   python {platform.python_version()} on {sys.platform}")
    print(f"  database {'given' if args.database_url else 'none -- first use not exercised'}")
    print()

    work = Path(tempfile.mkdtemp(prefix="chat_rag-dist-"))
    report = Report()
    try:
        artifacts = build(report, work / "dist", sdist=args.sdist)
        if not artifacts:
            return report.finish()

        for kind, artifact in artifacts.items():
            python = install(report, f"{kind}.install", work / kind, str(artifact))
            if python is not None:
                probe(report, python, expect_heavy=False,
                      database_url=args.database_url if kind == "wheel" else None,
                      prefix=kind)

        if args.with_extras:
            extras = install(report, "extras.install", work / "full",
                             f"{artifacts['wheel']}[all]")
            if extras is not None:
                probe(report, extras, expect_heavy=True, prefix="extras")
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
