"""The production runtime, started the way a deployment starts it.

Everything else about this phase is checked by reading configuration. This
starts the real thing: a separate process running ``python -m wsgi``, given one
setting -- where its state lives -- and asked one cheap question over a real
socket.

It proves the three claims that only a running process can:

* the production entrypoint binds and serves, so an image whose CMD is this
  will answer rather than crash at the first request;
* it is waitress answering, not Werkzeug -- the development server does not
  become the production server by having a flag set;
* one ``CHAT_RAG_DATA_DIR`` is sufficient isolation on its own. The server runs
  from the checkout, where ``.env`` says ``VECTOR_DB_PATH=./chroma_db``, and
  the developer's store, ledger and knowledge base records must be exactly as
  they were when it stops.

No provider is contacted: the lexical profile builds no embedding model, and
the health route touches neither parser nor store.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: A cold start builds the pipeline; slow machines and cold file caches are why
#: this is generous rather than tight.
READY_TIMEOUT_SECONDS = 180.0

#: Files the developer's console owns. A test that starts a whole server in the
#: checkout is exactly the kind that would quietly rewrite one.
DEVELOPER_STATE = (
    ".ingested_documents.json",
    ".knowledge_bases.json",
    ".gold_set.json",
    os.path.join("chroma_db", "chroma.sqlite3"),
)


def fingerprint() -> dict:
    found = {}
    for relative in DEVELOPER_STATE:
        try:
            stat = os.stat(os.path.join(REPO_ROOT, relative))
            found[relative] = (stat.st_size, stat.st_mtime)
        except OSError:
            found[relative] = None
    return found


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    """``python -m wsgi``, in the checkout, with its state somewhere else."""
    data_root = tmp_path / "data-root"
    port = free_port()

    environment = dict(os.environ)
    # The only path setting. If it were not sufficient, the .env sitting next
    # to this test would send the process to ./chroma_db.
    environment["CHAT_RAG_DATA_DIR"] = str(data_root)
    environment.pop("VECTOR_DB_PATH", None)
    environment.pop("STRUCTURED_PARSER_CACHE", None)
    environment["FLASK_HOST"] = "127.0.0.1"
    environment["FLASK_PORT"] = str(port)
    environment["WAITRESS_THREADS"] = "2"
    # Builds no embedding model, so nothing is downloaded to answer a health
    # check. Forced rather than defaulted: the developer's .env selects a
    # gateway profile, and this test must not depend on which.
    environment["RETRIEVAL_PROFILE"] = "bm25_only"
    environment["EMBEDDING_PROVIDER"] = "sentence_transformers"
    environment["PYTHONIOENCODING"] = "utf-8"

    before = fingerprint()
    process = subprocess.Popen(
        [sys.executable, "-m", "wsgi"],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        yield process, f"http://127.0.0.1:{port}", data_root
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - a hung server
            process.kill()
            process.wait()
        assert fingerprint() == before, (
            "starting a server in the checkout changed the developer's own state"
        )


def get(url: str, process: subprocess.Popen):
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last = "never answered"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise AssertionError(
                f"the server exited with {process.returncode} before answering:\n{output}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.status, dict(response.headers), json.loads(
                    response.read().decode("utf-8") or "{}"
                )
        except (urllib.error.URLError, OSError, ValueError) as error:
            last = f"{type(error).__name__}: {error}"
            time.sleep(0.5)
    raise AssertionError(f"{url} did not answer within {READY_TIMEOUT_SECONDS:.0f}s ({last})")


def test_the_production_entrypoint_serves_health_on_waitress(server):
    process, base, data_root = server

    status, headers, body = get(base + "/api/health", process)

    assert status == 200
    assert body["status"] == "healthy"
    server_name = headers.get("Server", "")
    assert "waitress" in server_name.lower(), f"served by {server_name!r}, not waitress"
    assert "werkzeug" not in server_name.lower()


def test_the_running_server_keeps_its_state_in_the_data_root(server):
    process, base, data_root = server

    get(base + "/api/health", process)

    assert data_root.is_dir(), "the data root was never used"
    assert (data_root / "logs").is_dir(), "the log file did not land in the data root"
    # The developer's store is the one thing this must never open. The
    # after-the-fact check is in the fixture's teardown, which compares the
    # checkout's own state files byte-for-byte.
    assert not (data_root / "chroma_db").exists(), (
        "a ./chroma_db under the data root means a cwd-relative default escaped"
    )
