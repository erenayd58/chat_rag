"""Can a burst of synchronous uploads lock the server out of answering anything?

This is not a simulation. It starts the production server -- waitress, the
real WSGI app, a real socket -- with two request threads, and asks the
question the compatibility path raises: while ``INGEST_SYNC_WAIT`` seconds of
synchronous uploads are in progress, can ``/api/health`` and job status still
be served?

Both answers are here. The first test removes the rationing and shows the
failure: two blocking uploads take both request threads and health goes
unanswered. The second leaves the product's own setting in place and shows
that it does not happen, because an upload that cannot get a waiting slot is
answered 202 with its job instead of holding a thread. The jobs themselves
are never rationed -- both uploads are accepted and both run -- so the
compatibility path degrades into the asynchronous one rather than taking the
server down with it.

Nothing here reaches a provider: the pipeline is a stub that blocks on an
event the test controls.
"""

from __future__ import annotations

import io
import json
import socket
import threading
import urllib.error
import urllib.request
import uuid
from types import SimpleNamespace

import pytest

import app as flask_app
from application import ingest as app_ingest
from application import workspace as app_workspace
import tempfile
from components.ingest import IngestManager
from components.knowledgebase.manager import KnowledgeBaseManager
from config.ingest import IngestLimits

THREADS = 2
#: Long enough that a blocked request is unambiguously blocked, short enough
#: that a mistake in this file cannot hang the suite for long.
SYNC_WAIT = 30.0
HEALTH_TIMEOUT = 3.0


class Chunker:
    last_canonical_units = None
    last_deep_result = None

    def get_name(self):
        return "StructuralChunker"

    def chunk_text_deep(self, *a, **k):  # pragma: no cover - presence is the contract
        raise AssertionError


class BlockingPipeline:
    """An ingest that runs until the test lets it finish."""

    def __init__(self):
        self.settings = SimpleNamespace(deep_analysis_api_key_env="FAKE_DEEP_KEY")
        self.chunker = Chunker()
        self.vector_db = SimpleNamespace(delete_by_doc_id=lambda d: None, get_all_chunks=lambda: [])
        self.last_deep_analysis_report = None
        self.last_parse_seconds = 0.1
        self.release = threading.Event()
        self.running = threading.Semaphore(0)

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        self.running.release()
        self.release.wait(60)
        return [SimpleNamespace(doc_id=f"doc-{uuid.uuid4().hex[:6]}")]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def server(tmp_path, monkeypatch):
    """The production server, in this process, with two request threads."""
    from waitress import create_server

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    kb_manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(flask_app.services, "kb_manager", kb_manager)
    monkeypatch.setattr(app_workspace, "stage_analysis", lambda *a, **k: {"status": "queued"})
    monkeypatch.setattr(flask_app.settings, "ingest_sync_wait", SYNC_WAIT)

    pipeline = BlockingPipeline()
    monkeypatch.setattr(flask_app.services, "get_pipeline", lambda *a, **k: pipeline)

    # One worker, room to queue: the second upload waits for the first, which
    # is what keeps both requests blocked long enough to ask the question.
    manager = IngestManager(
        IngestLimits(workers=1, queue_capacity=4, job_timeout_seconds=120),
        execute=lambda job: app_ingest.execute_job(flask_app.services, job),
    )
    monkeypatch.setattr(flask_app.services, "ingest_jobs", manager)

    port = free_port()
    instance = create_server(flask_app.app, host="127.0.0.1", port=port, threads=THREADS)
    thread = threading.Thread(target=instance.run, name="waitress-under-test", daemon=True)
    thread.start()
    kb = kb_manager.create("starvation-kb", chunker={"type": "structure_first"})

    yield SimpleNamespace(port=port, pipeline=pipeline, manager=manager, kb_id=kb["kb_id"])

    pipeline.release.set()
    manager.close(timeout=30)
    instance.close()
    thread.join(10)


def upload(port, kb_id, content: bytes, *, timeout=60.0):
    """A synchronous upload: no async flag, exactly as an older client sends."""
    boundary = uuid.uuid4().hex
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"kb_id\"\r\n\r\n{kb_id}\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{content.decode()}.txt\"\r\nContent-Type: text/plain\r\n\r\n",
    ]
    body = (parts[0] + parts[1]).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/documents/upload", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode())


def health(port, timeout=HEALTH_TIMEOUT):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as answer:
        return answer.status, json.loads(answer.read().decode())


def start_upload(server, content: bytes) -> tuple[threading.Thread, list]:
    outcome: list = []
    thread = threading.Thread(
        target=lambda: outcome.append(upload(server.port, server.kb_id, content)),
        daemon=True,
    )
    thread.start()
    return thread, outcome


def test_without_rationing_two_synchronous_uploads_take_the_whole_server(server, monkeypatch):
    """The failure this closes, demonstrated on the real server.

    With as many synchronous waiters allowed as there are request threads,
    two uploads occupy both and nothing else is served -- not health, not the
    job status a browser would be polling.
    """
    monkeypatch.setattr(flask_app.services, "sync_waiters", threading.BoundedSemaphore(THREADS))
    first, _ = start_upload(server, b"one")
    assert server.pipeline.running.acquire(timeout=20), "the first upload is being ingested"
    second, _ = start_upload(server, b"two")
    # The second request is queued behind the first *inside the route*: it
    # holds the other request thread while it waits for its job.
    assert server.manager.drain(0.5) is False

    with pytest.raises((urllib.error.URLError, socket.timeout, TimeoutError)):
        health(server.port)

    server.pipeline.release.set()
    first.join(30)
    second.join(30)
    # And once they finish, the server answers again: the threads were held,
    # not broken.
    assert health(server.port)[0] == 200


def test_with_the_products_rationing_health_and_status_are_always_served(server):
    """The fix: only some request threads may block on a job. The upload that
    cannot is still accepted and still runs -- it is answered 202."""
    assert flask_app.settings.ingest_limits.sync_waiters >= 1
    with_one_waiter = threading.BoundedSemaphore(1)
    original = flask_app.services.sync_waiters
    flask_app.services.sync_waiters = with_one_waiter
    try:
        blocking, _ = start_upload(server, b"one")
        assert server.pipeline.running.acquire(timeout=20)

        # The second synchronous upload finds no waiting slot. It returns at
        # once with its job rather than holding the last request thread.
        status, body = upload(server.port, server.kb_id, b"two", timeout=20)
        assert status == 202
        assert body["pending"] is True and body["busy"] is True
        assert body["job"]["status"] in ("queued", "running")

        # Which is the point: the server still answers everything else.
        code, payload = health(server.port)
        assert code == 200
        assert payload["ingest"]["running"] == 1
        assert payload["ingest"]["queued"] == 1

        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.port}/api/ingest/jobs?kb_id={server.kb_id}",
            timeout=HEALTH_TIMEOUT,
        ) as answer:
            assert answer.status == 200
            assert len(json.loads(answer.read().decode())["jobs"]) == 2

        job_id = body["job_id"]
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.port}/api/ingest/jobs/{job_id}", timeout=HEALTH_TIMEOUT
        ) as answer:
            assert answer.status == 200
            assert json.loads(answer.read().decode())["job"]["job_id"] == job_id

        server.pipeline.release.set()
        blocking.join(30)
        assert server.manager.drain(30)
        # Both uploads were ingested: rationing the waiting never rationed
        # the work.
        assert server.manager.stats["succeeded"] == 2
        assert server.manager.stats["rejected"] == 0
    finally:
        flask_app.services.sync_waiters = original
        server.pipeline.release.set()


def test_the_default_is_derived_from_the_request_thread_count():
    """Half the threads, so status traffic always has somewhere to land."""
    from config.ingest import limits_from_env

    assert limits_from_env({"WAITRESS_THREADS": "8"}).sync_waiters == 4
    assert limits_from_env({"WAITRESS_THREADS": "2"}).sync_waiters == 1
    assert limits_from_env({"WAITRESS_THREADS": "1"}).sync_waiters == 1
    assert limits_from_env({"INGEST_SYNC_WAITERS": "3", "WAITRESS_THREADS": "8"}).sync_waiters == 3
