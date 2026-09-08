"""Start the server, ask it one question, stop it.

    python tools/serve_smoke.py

``tools/import_smoke.py`` proves the declared dependencies can be imported.
This proves the next thing along: that the entrypoint a deployment actually
runs (``python -m asgi``) binds a socket, serves a request and stops when it is
asked to. It is what tells "the modules import" apart from "the container
serves", and it is cheap enough to run in the image build.

What it checks
--------------

* the entrypoint starts as its own process, exactly as the container's CMD
  starts it;
* ``/api/v1/health`` answers 200 over a real socket;
* the response is served by uvicorn, which is what proves the process that
  answered is the one the container runs;
* the process shuts down on the signal a service manager sends, and on POSIX
  it exits 0 rather than being killed (Windows has no graceful terminate to
  send, so there the exit status is not asserted).

The server is given two settings: ``CHAT_RAG_DATA_DIR``, pointing at a
throwaway directory, and ``DATABASE_URL``, inherited from the environment. The
first is the check hiding inside the check: if it were not sufficient on its
own -- if the checkout's ``.env`` could still supply a state path -- this
smoke would be writing into the developer's own checkout to answer a health
request.

The second cannot have a throwaway: the entrypoint refuses to serve without a
database, which is the behaviour this smoke would otherwise be proving by
accident. With ``DATABASE_URL`` unset the smoke says so and stops,
reporting success, because "no database here" is the state of an image build
and not a fault in the entrypoint. Wherever there *is* one -- a developer's
machine, CI -- the full check runs.

No model is downloaded and no provider is contacted: the lexical profile
builds no embedding model, and health touches neither parser nor store.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Long enough for a cold start that builds the pipeline on a slow machine.
READY_TIMEOUT_SECONDS = 180.0
STOP_TIMEOUT_SECONDS = 30.0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _environment(data_dir: str, port: int) -> dict:
    env = dict(os.environ)
    # One data root, and nothing else. The point of setting only this is that
    # it has to be sufficient on its own: config/paths.py derives the vector
    # store, the parser cache, the ledger and the logs from it, and the
    # checkout's .env is not allowed to move any of them once it is set.
    env["CHAT_RAG_DATA_DIR"] = data_dir
    # Inherited rather than invented: this smoke starts the real entrypoint,
    # and the real entrypoint refuses to serve without a reachable database.
    env["DATABASE_URL"] = os.environ.get("DATABASE_URL", "")
    env["FLASK_HOST"] = "127.0.0.1"
    env["FLASK_PORT"] = str(port)
    # bm25_only builds no embedding model, so no weights are downloaded to
    # answer a health check. Whether the product's dependencies satisfy its
    # imports has nothing to do with model files.
    env.setdefault("RETRIEVAL_PROFILE", "bm25_only")
    # PYTHONIOENCODING is deliberately *not* set. The server's output is a pipe
    # here, so on a Windows console whose code page is not UTF-8 this is the
    # case that used to crash the entrypoint before it bound. Setting it would
    # hide that; leaving it unset is what makes this smoke prove the start-up
    # path needs no help from whoever launched it.
    env.pop("PYTHONIOENCODING", None)
    return env


def _wait_for_health(url: str, process: subprocess.Popen) -> dict:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last_error = "never answered"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(
                f"the server exited with code {process.returncode} before answering {url}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8") or "{}")
                return {
                    "status": response.status,
                    "server": response.headers.get("Server", ""),
                    "body": body,
                }
        except (urllib.error.URLError, OSError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(0.5)
    raise SystemExit(f"{url} did not answer within {READY_TIMEOUT_SECONDS:.0f}s ({last_error})")


def main() -> int:
    if not (os.environ.get("DATABASE_URL") or "").strip():
        # The entrypoint refuses to serve without a database, by design. An
        # image build has none, so there is nothing here to prove or to fail.
        print("DATABASE_URL is not set: the entrypoint needs one, so there is "
              "no server to smoke here.")
        print("Set it (docker-compose.test.yml starts one) to run this check.")
        return 0

    data_dir = tempfile.mkdtemp(prefix="chat_rag-serve-smoke-")
    port = _free_port()
    url = f"http://127.0.0.1:{port}/api/v1/health"

    print(f"python  {sys.version.split()[0]}")
    print(f"state   {data_dir}  (throwaway; no real deployment is read or written)")
    print(f"serving python -m asgi on 127.0.0.1:{port}")

    process = subprocess.Popen(
        [sys.executable, "-m", "asgi"],
        cwd=ROOT,
        env=_environment(data_dir, port),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    failures: list[str] = []
    try:
        health = _wait_for_health(url, process)
        print(f"  health  {health['status']}  Server: {health['server'] or '(none)'}")

        if health["status"] != 200:
            failures.append(f"/api/v1/health answered {health['status']}")
        server_header = health["server"].lower()
        if "uvicorn" not in server_header:
            failures.append(
                f"served by {health['server'] or 'an unnamed server'}, not uvicorn "
                "-- the entrypoint that answered is not the one a deployment runs"
            )
        # The contract's own shape: a resource at the top level, no envelope.
        # ``ready`` is the field a load balancer acts on, and it stays true
        # while the service reports itself degraded or overloaded.
        if health["body"].get("ready") is not True:
            failures.append(f"health body says {health['body']!r}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            failures.append(f"did not stop within {STOP_TIMEOUT_SECONDS:.0f}s of being asked")
        output = process.stdout.read() if process.stdout else ""

    # Popen.terminate() is TerminateProcess on Windows -- a kill, with no
    # graceful equivalent to send from here -- so the exit status only means
    # something on POSIX, where it is the SIGTERM the container sends.
    if os.name != "nt" and process.returncode not in (0, None):
        failures.append(f"exited {process.returncode} on SIGTERM, expected a clean 0")

    if not os.path.isdir(data_dir):
        failures.append("the data directory vanished")
    print(f"  stopped exit={process.returncode}")

    shutil.rmtree(data_dir, ignore_errors=True)

    if failures:
        print()
        for failure in failures:
            print(f"  FAIL  {failure}")
        print()
        print("server output:")
        for line in (output or "").splitlines()[-40:]:
            print(f"  | {line}")
        return 1

    print()
    print("the entrypoint starts, serves /api/v1/health on uvicorn and stops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
