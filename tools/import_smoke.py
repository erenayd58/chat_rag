"""Import the product-facing modules and say which ``amsc`` answered.

Run this in a *clean* environment -- a fresh virtualenv built from
requirements.txt, or the Docker image -- to prove that the declared
dependency alone can start the application. Run in a development checkout it
proves something weaker but still useful: that the working tree imports.
Either way it names the ``amsc`` that was actually loaded and, when pip
recorded one, the exact revision it came from, so "it works on my machine"
becomes a checkable statement rather than a claim.

    python tools/import_smoke.py

Exit code 0 means every module imported. No model is downloaded, no provider
is contacted and no application state is written: importing is all it does.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile

#: Run as ``python tools/import_smoke.py``, sys.path[0] is tools/, not the
#: checkout. The application's own packages live one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: Imported in the order a request path loads them, most-depended-on first, so
#: a failure names the deepest module that broke rather than ``asgi``.
PRODUCT_MODULES = (
    "config.settings",
    "config.paths",
    "components.chunker.structural_chunker",
    "components.chunker.deep_analysis",
    "components.context.assembler",
    "components.retriever.hybrid_rrf_retriever",
    "components.parsers.structured_pdf_parser",
    "components.provenance",
    "components.viewer.methods",
    "components.viewer.analysis",
    "pipeline.rag_pipeline",
    "application.services",
    "application.ingest",
    "application.query",
    "interfaces.http.v1",
    "interfaces.http",
    "asgi",
)


def _amsc_origin() -> str:
    try:
        import amsc
    except Exception as error:  # noqa: BLE001
        return f"amsc could not be imported at all: {type(error).__name__}: {error}"
    where = getattr(amsc, "__file__", "?")
    try:
        import importlib.metadata as metadata

        direct = metadata.distribution("amsc-poc").read_text("direct_url.json")
        if direct:
            info = json.loads(direct)
            revision = (info.get("vcs_info") or {}).get("commit_id")
            editable = (info.get("dir_info") or {}).get("editable")
            if editable:
                return f"{where}\n  installed as an EDITABLE checkout of {info.get('url')}"
            if revision:
                return f"{where}\n  installed from {info.get('url')} at {revision}"
        return f"{where}\n  installed from a release (no direct_url record)"
    except Exception:  # noqa: BLE001
        return where


def main() -> int:
    # Importing ``asgi`` builds the default pipeline against whatever state
    # directory is configured. Point it at a throwaway one so a smoke check
    # never reads or writes a real deployment's knowledge bases.
    os.environ.setdefault("CHAT_RAG_DATA_DIR", os.path.join(tempfile.gettempdir(), "chat_rag-import-smoke"))
    # Importing ``asgi`` builds the default pipeline, and the default embedding
    # provider loads a sentence-transformers model in its constructor -- which
    # downloads one when the machine has none cached. Whether the declared
    # dependencies satisfy the product's imports has nothing to do with model
    # weights, so the smoke runs the one profile that builds no embedding
    # model at all. An operator who sets the variable still wins.
    os.environ.setdefault("RETRIEVAL_PROFILE", "bm25_only")
    # Nothing else is set. A data directory is now sufficient on its own:
    # config/paths.py derives the parser cache and every other file from it,
    # and a ``STRUCTURED_PARSER_CACHE`` left in a developer's .env is refused
    # rather than applied once a data root is declared. This script used to
    # have to force those by hand, which meant the isolation lived in the
    # smoke check instead of in the application. The vectors are not among
    # them at all any more: they are rows, reached by DATABASE_URL, and this
    # check never opens a database.
    data_dir = os.environ["CHAT_RAG_DATA_DIR"]

    print(f"python  {sys.version.split()[0]}")
    print(f"amsc    {_amsc_origin()}")
    print(f"state   {data_dir}  (throwaway; no real deployment is read or written)")

    from config import paths  # after the environment above, never before

    print(f"cache   {paths.canonical_cache()}")
    for line in paths.diagnostics():
        print(f"  note  {line}")
    for resolved in (paths.canonical_cache(), paths.viewer_live_analysis()):
        if not os.path.abspath(resolved).startswith(os.path.abspath(data_dir)):
            print(f"  FAIL  {resolved} is outside the throwaway data directory")
            return 1

    failures: list[tuple[str, str]] = []
    for name in PRODUCT_MODULES:
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001
            failures.append((name, f"{type(error).__name__}: {error}"))
            print(f"  FAIL  {name}: {type(error).__name__}: {error}")
        else:
            print(f"  ok    {name}")

    if failures:
        print(f"\n{len(failures)} of {len(PRODUCT_MODULES)} product modules could not be imported.")
        print("A missing amsc module or attribute means requirements.txt pins a revision "
              "older than the code that uses it.")
        return 1
    print(f"\nall {len(PRODUCT_MODULES)} product modules imported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
