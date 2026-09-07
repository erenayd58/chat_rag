"""The Flask application: construction, wiring, and the development entrypoint.

What is here is what belongs to *this framework* and this process -- the app
object, the session cookie's key, CORS, the container the routes are given,
and the development server. The product's behaviour is in :mod:`application`
and the HTTP translation of it is in :mod:`interfaces.http`; neither imports
this module, which is what lets the same application be served by something
other than Flask later.
"""
import os
import secrets

from flask import Flask
from flask_cors import CORS

import interfaces.http as http
from application.services import default_services
from runtime import bootstrap
from utils import get_logger

logger = get_logger("FlaskApp")


def _session_secret() -> str:
    """The key that signs the session cookie.

    ``FLASK_SECRET_KEY`` when it is set. Otherwise a fresh random key for this
    process: sessions then last as long as the process does, which is right for
    local development and honest in production, where the alternative used to
    be a constant published in this file -- a key everyone with the source
    could forge a session cookie with. The only thing the cookie carries is a
    per-browser ``session_id`` used to pick a cached pipeline, so a restart
    costs nothing but that.
    """
    configured = (os.getenv('FLASK_SECRET_KEY') or '').strip()
    if configured:
        return configured
    logger.warning(
        "FLASK_SECRET_KEY is not set; signing sessions with a key generated "
        "for this process. Sessions will not survive a restart and cannot be "
        "shared between instances. Set FLASK_SECRET_KEY for a deployment."
    )
    return secrets.token_hex(32)


logger.info("Initializing Flask application...")

#: The application, composed once for this process. Built before the Flask app
#: because the routes are handed it rather than reaching for it.
services = default_services()
settings = services.settings

app = Flask(__name__)
app.secret_key = _session_secret()
# The session cookie is never read by a script and never needs to travel on a
# cross-site request; both are Flask's own defaults, stated here so a future
# change to them is deliberate.
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
CORS(app)

http.register(app, services)


# ---- The entrypoints' shared start-up, bound to this process's container ----
enable_console_utf8 = bootstrap.enable_console_utf8
development_server_options = bootstrap.development_server_options


def startup_banner() -> None:
    bootstrap.startup_banner(services)


def resume_background_work() -> None:
    bootstrap.resume_background_work(services)


if __name__ == '__main__':
    # THE DEVELOPMENT ENTRYPOINT.
    #
    # Werkzeug's server is not a production server, and this file does not
    # pretend otherwise: a deployment runs `python -m wsgi`, which serves this
    # same `app` object on waitress, in one process, with no debugger and no
    # reloader to switch off.
    enable_console_utf8()
    startup_banner()
    resume_background_work()

    options = development_server_options()

    print()
    print("=" * 80)
    print(f"Development server (Werkzeug, debug="
          f"{'on' if options['debug'] else 'off'}): "
          f"http://{options['host']}:{options['port']}")
    print("Production: python -m wsgi")
    print("=" * 80)
    print()

    app.run(**options)
