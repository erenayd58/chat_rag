"""Two entrypoints, and which one a deployment reaches.

``python app.py`` is the development server: Werkzeug, the reloader, the
interactive debugger, loopback. It is convenient and it is not a production
runtime -- and until this phase it was also the container's CMD, so production
*was* the development server with a flag turned off, and stayed that way for as
long as everyone remembered the flag.

``python -m wsgi`` is the production server: waitress, one process, a bounded
pool of request threads, no debugger in the image at all. These tests pin the
difference, and the two things that must not drift: the production entrypoint
must never serve with debug, and both must serve the same application object.

Why one process is in the module docstring of ``wsgi.py``; the short version is
that the Viewer packaging worker is one thread over an in-memory queue and the
pipeline cache is a module global, so a second process would duplicate both.
"""

from __future__ import annotations

import os

import pytest

import app as flask_app
import wsgi


@pytest.fixture(autouse=True)
def preserve_signal_handlers():
    """Installing real handlers is the point of one test and a side effect of
    another; neither may leave pytest's own process changed."""
    import signal

    names = [n for n in ("SIGTERM", "SIGINT") if hasattr(signal, n)]
    saved = {n: signal.getsignal(getattr(signal, n)) for n in names}
    yield
    for name, handler in saved.items():
        try:
            signal.signal(getattr(signal, name), handler)
        except (ValueError, TypeError, OSError):  # pragma: no cover
            pass


@pytest.fixture
def clean_server_env(monkeypatch):
    for name in ("FLASK_HOST", "FLASK_PORT", "FLASK_DEBUG", "WAITRESS_THREADS",
                 "WAITRESS_CHANNEL_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------ the two servers


def test_the_production_entrypoint_serves_the_same_application():
    assert wsgi.application is flask_app.app


def test_an_imported_application_is_never_in_debug():
    """Importing the app must not turn anything on by itself."""
    assert flask_app.app.debug is False


def test_the_production_server_has_no_debug_setting_to_get_wrong(clean_server_env, monkeypatch):
    """Not "debug defaults to off" -- there is no debug switch on this path.

    FLASK_DEBUG=true, the developer default, must not reach the production
    server: waitress does not run Werkzeug's debugger, and nothing in
    ``server_options`` consults the variable.
    """
    monkeypatch.setenv("FLASK_DEBUG", "true")

    options = wsgi.server_options()

    assert "debug" not in options
    assert not any("debug" in str(value).lower() for value in options.values())


# ------------------------------------------------------------------ defaults


def test_production_defaults(clean_server_env):
    options = wsgi.server_options()

    assert options["host"] == "0.0.0.0", "a container's port mapping needs every interface"
    assert options["port"] == 5005
    assert options["threads"] == 8
    assert options["clear_untrusted_proxy_headers"] is True


def test_production_options_are_configurable(clean_server_env, monkeypatch):
    monkeypatch.setenv("FLASK_HOST", "127.0.0.1")
    monkeypatch.setenv("FLASK_PORT", "9001")
    monkeypatch.setenv("WAITRESS_THREADS", "2")

    options = wsgi.server_options()

    assert (options["host"], options["port"], options["threads"]) == ("127.0.0.1", 9001, 2)


@pytest.mark.parametrize("bad", ["", "   ", "0", "-4", "eight"])
def test_a_nonsense_thread_count_falls_back_rather_than_crashing(
    clean_server_env, monkeypatch, bad
):
    """A typo in an env file must not leave a deployment with zero threads."""
    monkeypatch.setenv("WAITRESS_THREADS", bad)

    assert wsgi.server_options()["threads"] == 8


def test_development_defaults_to_loopback_with_debug(clean_server_env):
    """The developer keeps the reloader; the network does not get the debugger."""
    options = flask_app.development_server_options()

    assert options["host"] == "127.0.0.1", "0.0.0.0 exposed the debugger to the LAN"
    assert options["port"] == 5005
    assert options["debug"] is True


@pytest.mark.parametrize("value,expected", [
    ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ("true", True), ("1", True), ("anything else", True),
])
def test_the_development_server_still_honours_flask_debug(
    clean_server_env, monkeypatch, value, expected
):
    monkeypatch.setenv("FLASK_DEBUG", value)
    assert flask_app.development_server_options()["debug"] is expected


def test_the_development_host_can_still_be_opened_deliberately(clean_server_env, monkeypatch):
    monkeypatch.setenv("FLASK_HOST", "0.0.0.0")
    assert flask_app.development_server_options()["host"] == "0.0.0.0"


# --------------------------------------------------------------- session key


def test_the_session_key_is_not_the_one_printed_in_the_source():
    """It used to default to a constant literal in app.py.

    Anyone with the source could forge a session cookie signed with it, and
    every deployment that did not set the variable shared the same one.
    """
    assert flask_app.app.secret_key
    assert flask_app.app.secret_key != 'your-secret-key-change-in-production'


def test_a_configured_session_key_is_used(monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "  a-real-deployment-key  ")
    assert flask_app._session_secret() == "a-real-deployment-key"


def test_an_absent_session_key_is_random_per_process(monkeypatch):
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    assert flask_app._session_secret() != flask_app._session_secret()


def test_the_session_cookie_is_not_readable_by_a_script():
    assert flask_app.app.config["SESSION_COOKIE_HTTPONLY"] is True
    assert flask_app.app.config["SESSION_COOKIE_SAMESITE"] == "Lax"


# --------------------------------------------------------- what main() runs


def test_main_starts_waitress_and_stops_cleanly(monkeypatch, clean_server_env, tmp_path):
    """No socket is bound here; this pins how ``main`` wires the server up."""
    recorded = {}

    class FakeServer:
        def __init__(self):
            self.closed = False

        def run(self):
            recorded["ran"] = True

        def close(self):
            self.closed = True

    fake = FakeServer()

    def create_server(application, **kwargs):
        recorded["application"] = application
        recorded["kwargs"] = kwargs
        return fake

    import waitress

    monkeypatch.setattr(waitress, "create_server", create_server)
    monkeypatch.setattr(flask_app, "startup_banner", lambda: recorded.setdefault("banner", True))
    monkeypatch.setattr(
        flask_app, "resume_background_work", lambda: recorded.setdefault("resumed", True)
    )

    assert wsgi.main() == 0

    assert recorded["application"] is flask_app.app
    assert recorded["kwargs"] == wsgi.server_options()
    assert recorded["ran"] is True
    assert fake.closed is True, "the listening socket has to be released on the way out"
    # A restart has to pick up an interrupted Viewer packaging job, and that
    # used to happen only under `python app.py`.
    assert recorded["resumed"] is True
    assert recorded["banner"] is True


def test_a_sigterm_handler_is_installed_where_the_platform_has_one():
    """``docker stop`` sends SIGTERM, and Python's default for it is to die."""
    import signal

    if not hasattr(signal, "SIGTERM"):  # pragma: no cover - platform without it
        pytest.skip("no SIGTERM on this platform")

    wsgi._install_shutdown_handlers()

    installed = signal.getsignal(signal.SIGTERM)
    assert callable(installed)
    with pytest.raises(SystemExit):
        installed(signal.SIGTERM, None)


# ------------------------------------------------------------- the container


def test_the_image_runs_the_production_entrypoint():
    """The Dockerfile's CMD, read as the deployment contract it is."""
    dockerfile = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "Dockerfile"), encoding="utf-8"
    ).read()

    assert 'CMD ["python", "-m", "wsgi"]' in dockerfile
    assert 'CMD ["python", "app.py"]' not in dockerfile
