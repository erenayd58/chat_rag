"""The deployment is a file, so it is read like one.

``docker compose up --build`` on a clean machine has to produce a working
system, and every way that has failed here was a statement in a tracked file
being wrong rather than code being wrong:

* the console's backend address was resolved by ``next build`` and frozen into
  the image, so configuring it at run time did nothing;
* the schema was a manual step in a comment, so a fresh stack came up against
  an empty database and every screen 500'd;
* a container started before the thing it depends on was ready and crash-looped
  until it happened to win.

None of those is visible to a unit test of the application. They are visible
here, because each is a line in ``docker-compose.yml``, ``Dockerfile``,
``frontend/Dockerfile``, ``docker-entrypoint.sh`` or ``next.config.mjs``, and a
line can be read.

This file does not build or run anything. Whether the stack *actually* comes up
is the reproducibility gate's question (``tools/verify_reproducibility.py``);
this is the cheap half that runs on every commit and catches the wrong
statement before an eight-minute build does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
COMPOSE = REPO / "docker-compose.yml"
FRONTEND = REPO / "frontend"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _instructions(path: Path) -> str:
    """A Dockerfile or a JavaScript config with its prose removed.

    These files carry long comments explaining what they deliberately do *not*
    do, so a test that searched the whole text would be answered by the very
    sentence saying the thing is gone.
    """
    lines = []
    in_block = False
    for line in _read(path).splitlines():
        stripped = line.strip()
        if in_block:
            if "*/" in stripped:
                in_block = False
            continue
        if stripped.startswith("/*"):
            in_block = "*/" not in stripped
            continue
        if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("*"):
            continue
        lines.append(line)
    return "\n".join(lines)


@pytest.fixture(scope="module")
def stack() -> dict:
    return yaml.safe_load(_read(COMPOSE))


@pytest.fixture(scope="module")
def services(stack) -> dict:
    return stack["services"]


# ------------------------------------------------------------ what is there


def test_the_stack_is_the_three_services_the_product_is(services):
    """Database, application, console. A fourth would be a piece of the
    architecture the documentation does not describe."""
    assert set(services) == {"db", "app", "frontend"}


def test_the_database_is_the_image_that_can_create_the_vector_extension(services):
    """A stock ``postgres`` image cannot, and the migration that creates it
    would fail on the first ``docker compose up`` of a fresh volume."""
    assert services["db"]["image"].startswith("pgvector/pgvector:")


def test_both_application_images_are_built_from_this_repository(services):
    """Neither is pulled: what runs is what this checkout describes."""
    assert services["app"]["build"]["context"] == "."
    assert services["frontend"]["build"]["context"] == "./frontend"


# ------------------------------------------------------------------ ordering


def _condition(service: dict, upstream: str) -> str:
    return service["depends_on"][upstream]["condition"]


def test_the_application_waits_for_a_healthy_database(services):
    """``service_started`` would mean "the process exists", which for
    PostgreSQL is true several seconds before it accepts a connection. The
    application refuses to serve without one, so that difference is a container
    that exits and restarts until the timing happens to work."""
    assert _condition(services["app"], "db") == "service_healthy"


def test_the_console_waits_for_a_healthy_application(services):
    """The console's first screen lists knowledge bases. Up before the
    application is up means a first paint that is an error, and it also means
    the console is serving before the migration in front of the application has
    run."""
    assert _condition(services["frontend"], "app") == "service_healthy"


@pytest.mark.parametrize("name", ["db", "app", "frontend"])
def test_every_service_says_how_to_tell_whether_it_is_well(services, name):
    """A ``depends_on: service_healthy`` is only as good as the check it waits
    on, and a service with no check cannot be waited on at all."""
    assert "healthcheck" in services[name], f"{name} has no health check"
    assert services[name]["healthcheck"].get("test")


def test_the_application_health_check_is_the_contract_route(services):
    """Not a port scan and not a page: ``/api/v1/health`` is the route that
    answers while degraded, loads no model and touches no store."""
    probe = " ".join(str(part) for part in services["app"]["healthcheck"]["test"])
    assert "/api/v1/health" in probe


@pytest.mark.parametrize("name", ["db", "app", "frontend"])
def test_every_service_comes_back_after_a_restart(services, name):
    assert services[name]["restart"] == "unless-stopped"


def test_the_application_is_given_time_to_drain(services):
    """The lifespan's shutdown lets the ingest jobs already running finish and
    then returns the database pool. A grace period shorter than that turns a
    graceful stop into a kill, which loses the one write a job must not lose --
    its final ledger row."""
    assert services["app"]["stop_grace_period"] == "15s"


# ------------------------------------------------- where the backend lives


def test_the_console_is_told_where_the_application_is_by_service_name(services):
    """Not localhost. Inside a container ``127.0.0.1`` is that container, so a
    console configured that way reaches nothing while the application it wants
    is healthy on the same network."""
    address = services["frontend"]["environment"]["CHAT_RAG_API_URL"]
    assert address == "http://app:5005"
    assert "localhost" not in address and "127.0.0.1" not in address


def test_the_console_resolves_that_address_at_run_time_not_build_time():
    """The regression this whole file exists for.

    ``next.config.mjs`` used to return a ``rewrites()`` entry built from
    ``CHAT_RAG_API_URL``. Next.js resolves ``rewrites()`` during ``next build``
    and writes the destination into ``.next/routes-manifest.json``, so the
    image carried the developer default and the environment variable on the
    container was read by nothing. The forwarding is a route handler now.
    """
    config = _instructions(FRONTEND / "next.config.mjs")
    assert "rewrites" not in config, (
        "the backend address is resolved at build time again; "
        "frontend/lib/api/proxy.ts explains why that cannot work"
    )

    proxy = _instructions(FRONTEND / "lib" / "api" / "proxy.ts")
    assert "process.env.CHAT_RAG_API_URL" in proxy
    assert (FRONTEND / "app" / "api" / "v1" / "[...path]" / "route.ts").is_file()


def test_the_console_image_takes_no_backend_address_as_a_build_argument():
    """The other half of the same rule: an image that had to be told at build
    time where the application is would be an image per environment."""
    dockerfile = _instructions(FRONTEND / "Dockerfile")
    for line in dockerfile.splitlines():
        if line.strip().startswith("ARG "):
            assert "CHAT_RAG_API_URL" not in line, line


# --------------------------------------------------------------- the schema


def test_the_container_migrates_before_it_serves():
    """Automatic, and still a migration: nothing in the application creates a
    table, and the entrypoint runs the tool that takes a lock and reports the
    revisions."""
    entrypoint = _instructions(REPO / "docker-entrypoint.sh")
    assert "python -m tools.migrate" in entrypoint

    dockerfile = _instructions(REPO / "Dockerfile")
    assert 'ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]' in dockerfile
    # The command stays what it was: an entrypoint that replaced it would be a
    # second way to start the product.
    assert 'CMD ["python", "-m", "asgi"]' in dockerfile


def test_the_entrypoint_hands_the_signal_to_the_server():
    """``exec``, so uvicorn is PID 1 and takes the SIGTERM ``docker stop``
    sends. A shell left in front of it would absorb the signal and the server
    would be killed at the end of the grace period instead of draining."""
    entrypoint = _read(REPO / "docker-entrypoint.sh")
    assert re.search(r'^exec "\$@"$', entrypoint, re.M)


def test_the_entrypoint_has_unix_line_endings():
    """A CRLF shebang makes the kernel look for ``/bin/sh\\r``, and the
    container exits reporting that the *interpreter* does not exist -- which
    sends everyone to look at the script. ``.gitattributes`` pins it; this is
    what notices when that stops being true."""
    assert b"\r\n" not in (REPO / "docker-entrypoint.sh").read_bytes()


# The premise this migration story rests on -- that no source file builds a
# schema by itself -- is real, and it is asserted in exactly one place:
# tests/storage/test_migrations.py, which reads the tracked files and keeps the
# allow-list. It was briefly stated here as well, and the duplicate is the
# whole argument against duplicating it: a second copy of the rule contains the
# very string the first one searches for, so committing this file made the
# original fail. One owner per invariant.


# --------------------------------------------- the fixture is not the product


def test_the_test_database_is_a_different_compose_project_from_the_deployment():
    """Both files describe a service called `db`, and compose derives a project
    name from the directory -- which is the same directory. Without a project
    name of its own, `docker compose -f docker-compose.test.yml up` is read as
    "the chat_rag project's db service is this now" and **replaces the running
    deployment's database with the test fixture**. The application container
    stays up and healthy, still resolving the hostname `db`, now pointing at a
    server that has no `chat_rag` database in it.

    That is not hypothetical: it happened while Step 14 was being verified, and
    the only symptom was a stack that had been working a minute earlier.
    """
    fixture = yaml.safe_load(_read(REPO / "docker-compose.test.yml"))
    stack = yaml.safe_load(_read(COMPOSE))

    assert fixture.get("name"), (
        "docker-compose.test.yml has no project name, so it shares the "
        "deployment's and its `db` service silently replaces the deployment's"
    )
    assert fixture["name"] != stack.get("name")
    # And the service names really do collide, which is what makes the project
    # name load-bearing rather than tidy.
    assert set(fixture["services"]) & set(stack["services"])


def test_the_two_stacks_cannot_contend_for_a_port():
    """The fixture is on 55432 so it misses both the deployment's database
    (unpublished) and any PostgreSQL already installed on the machine."""
    fixture = yaml.safe_load(_read(REPO / "docker-compose.test.yml"))
    published = {
        str(mapping).split(":")[0]
        for service in fixture["services"].values()
        for mapping in service.get("ports", [])
    }
    assert published == {"55432"}


# ------------------------------------------------------- writable state


def _mounts(service: dict) -> list[str]:
    return [str(entry) for entry in service.get("volumes", [])]


@pytest.mark.parametrize("name", ["db", "app", "frontend"])
def test_no_service_keeps_writable_state_on_a_host_path(services, name):
    """Every mount is a Docker-managed named volume. Not style -- ownership.

    These were `./.docker-data/...` bind mounts, and on Docker Desktop that
    works: its filesystem translation layer presents a bind mount as owned by
    whoever asks. On Linux, Docker creates a missing bind-mount source as
    root:root and the mount carries that ownership into the container. The
    application image drops to uid 10001 before it runs anything, so on the
    first clean Linux runner the first thing it did was

        PermissionError: [Errno 13] Permission denied: '/data/logs'

    A named volume is initialised from the image's content at the mount point,
    ownership included, so the directory arrives owned by the user that has to
    write it -- on every platform, and with no chmod anywhere.

    A bind mount is spelled with a path separator in its source; a named volume
    is a bare name.
    """
    offenders = [entry for entry in _mounts(services[name])
                 if entry.startswith((".", "/", "~", "$")) or ":" not in entry]
    assert offenders == [], (
        f"{name} keeps state on a host path: {offenders}. On Linux that "
        "directory is root-owned and the container cannot write to it."
    )


def test_every_volume_the_services_use_is_declared(stack, services):
    """A named volume compose was never told about is a typo that silently
    becomes an anonymous volume -- which is not shared, not reset by
    `down -v` by name, and gone on the next `up --force-recreate`."""
    declared = set(stack.get("volumes") or {})
    used = {entry.split(":")[0]
            for name in services for entry in _mounts(services[name])}
    assert used <= declared, f"undeclared volumes: {sorted(used - declared)}"


def test_the_image_creates_every_directory_a_volume_is_mounted_over():
    """The other half of the fix, and the half that is easy to lose.

    A named volume inherits the image's ownership at its mount point *only
    where that path exists in the image*. Where it does not, Docker creates it
    as root:root and the permission error is back, now with a named volume to
    make it look impossible. `artifacts/runs` and `artifacts/reports` are
    exactly that case: .dockerignore keeps generated evaluation output out of
    the image, so nothing else would create them.

    So every mount point the application container uses has to be named in the
    Dockerfile's mkdir, and owned by `app`.
    """
    dockerfile = _instructions(REPO / "Dockerfile")
    stack = yaml.safe_load(_read(COMPOSE))
    targets = [entry.split(":")[1]
               for entry in _mounts(stack["services"]["app"])]
    assert targets, "the application container mounts nothing at all"

    created = " ".join(line for line in dockerfile.splitlines() if "mkdir" in line)
    missing = [target for target in targets if target not in created]
    assert missing == [], (
        f"the image never creates {missing}, so Docker will create the "
        "mount point as root and uid 10001 cannot write to it"
    )
    assert re.search(r"chown -R app:app.*/data", dockerfile), (
        "the created directories are not given to the user that runs"
    )


def test_the_database_volume_is_not_inside_the_application_one(services):
    """They were: `./.docker-data` was mounted at /data *and* held
    `postgres/`, so the application container could read the database's files.
    Separate volumes are separate concerns, and neither has to know the
    other's layout."""
    app_targets = {entry.split(":")[1] for entry in _mounts(services["app"])}
    db_sources = {entry.split(":")[0] for entry in _mounts(services["db"])}
    app_sources = {entry.split(":")[0] for entry in _mounts(services["app"])}
    assert db_sources.isdisjoint(app_sources)
    assert "/var/lib/postgresql/data" not in app_targets


# ------------------------------------------------------------------ secrets


def test_no_tracked_deployment_file_carries_a_real_credential(services):
    """The database password is a compose default because the database has no
    published port: it guards a network only these containers are on. Anything
    stronger belongs in an untracked override, and the file has to keep saying
    so."""
    assert "ports" not in services["db"], (
        "the database is published; its password is now a real secret and "
        "cannot be a default in a tracked file"
    )
    password = services["db"]["environment"]["POSTGRES_PASSWORD"]
    assert password.startswith("${POSTGRES_PASSWORD"), (
        "the password must be overridable from the environment"
    )


def test_the_private_override_file_is_not_tracked():
    """``.env.docker`` is tracked and secret-free; ``.env.docker.local`` is
    where a key goes, and git must not be able to see it."""
    ignored = _read(REPO / ".gitignore")
    assert ".env.docker.local" in ignored

    tracked = _read(REPO / ".dockerignore")
    assert ".env" in tracked and "!.env.docker" in tracked


# -------------------------------------------------------- image hygiene


def test_the_application_image_does_not_carry_the_console():
    """Two images, two build contexts. Copying ``frontend/`` into the
    application image would put a host-built ``node_modules`` -- the wrong
    platform's binaries -- into a layer that never runs it."""
    assert "frontend/" in _instructions(REPO / ".dockerignore")


def test_the_console_image_installs_from_the_lockfile():
    """``npm ci`` fails on a ``package.json`` the lockfile does not match;
    ``npm install`` quietly resolves something newer, which is how two builds
    of one commit stop being the same build."""
    dockerfile = _instructions(FRONTEND / "Dockerfile")
    assert "npm ci" in dockerfile
    assert "npm install" not in dockerfile


def test_the_console_image_ships_only_what_the_server_reaches():
    """``output: 'standalone'`` traces the modules the server actually uses and
    copies those; without it the runtime stage would carry the whole install,
    dev dependencies included."""
    assert "output: 'standalone'" in _instructions(FRONTEND / "next.config.mjs")
    assert "/app/.next/standalone" in _instructions(FRONTEND / "Dockerfile")


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "frontend/Dockerfile"])
def test_neither_image_runs_as_root(dockerfile):
    """A container that does not need to be root is one fewer thing that a
    remote code execution reaches."""
    assert re.search(r"^USER app$", _instructions(REPO / dockerfile), re.M), dockerfile


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "frontend/Dockerfile"])
def test_neither_image_floats_on_a_tag_that_moves(dockerfile):
    """``FROM node:22`` builds a different runtime next month. Every base is
    pinned to something that does not move under a rebuild."""
    for line in _instructions(REPO / dockerfile).splitlines():
        if not line.startswith("FROM "):
            continue
        image = line.split()[1]
        assert ":" in image, f"{dockerfile}: {image} has no tag at all"
        tag = image.split(":", 1)[1]
        assert tag != "latest", f"{dockerfile}: {image} floats"
        assert re.search(r"\d+\.\d+", tag), (
            f"{dockerfile}: {image} is pinned no more precisely than a major "
            "version"
        )
