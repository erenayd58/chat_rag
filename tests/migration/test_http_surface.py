"""The HTTP surface, and what its refusals mean.

The console is Flask today and will not be. A framework port is proven by
parity, and parity needs a list of what there is to be at parity with --
which nothing in this repository had. Ninety tests drove individual routes;
none said how many routes there are, so a route lost in a port would be
found only if some unrelated test happened to use it, and a route quietly
added would never be found at all.

So this module pins the surface itself, in two halves.

**What exists.** The README's *console API* table is the published list, and
it is compared against the application's real routing table. Neither may
drift from the other: a new endpoint is documented in the same commit that
adds it, and a removal is a removal from both. That table is also the input
to the migration -- the set of paths and methods a FastAPI application has
to answer -- and to the cleanup before it, because a route no screen and no
client calls is visible here as a row nobody claims.

**What a refusal means.** The distinctions the product makes are a contract:
a malformed request, an unknown resource, a rejected payload and a server
fault are four different answers, and a client (the console's own
JavaScript, the Viewer's relay) branches on them. They are asserted here as
a taxonomy rather than one at a time, because the failure mode of a rewrite
is not losing one code -- it is collapsing several into 500 or into 400.

What this module does **not** pin: any error *message*, the internals of the
routing table, Flask's own behaviour (its automatic ``HEAD``/``OPTIONS``, its
static endpoint, its 405), or the JSON body of a success. Those belong to the
implementation and may all change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import app as flask_app

REPO = Path(__file__).resolve().parents[2]

#: The rendered screens. Not part of the API contract -- they are the ones
#: Next.js replaces -- but they are routes, so they are declared here rather
#: than left to make the comparison below fail.
PAGES = {
    ("GET", "/"),
    ("GET", "/chat"),
    ("GET", "/lab"),
    ("GET", "/kb/<kb_id>"),
}

#: Redirects kept so an old bookmark still works. Named here because "is this
#: still needed?" is a cleanup question with an answer, not a mystery.
LEGACY_REDIRECTS = {
    ("GET", "/documents"),   # -> /
    ("GET", "/chunks"),      # -> /lab
}

#: Flask's own endpoint, not the product's.
FRAMEWORK_ROUTES = {("GET", "/static/<path:filename>")}

#: Verbs Flask adds by itself. A port is free to add or not add them.
IMPLICIT_METHODS = {"HEAD", "OPTIONS"}


def _documented_api() -> set[tuple[str, str]]:
    """The endpoints the README's *console API* table publishes."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme.split("## The console API", 1)
    assert len(section) == 2, "the README no longer has a console API section"
    table = section[1].split("\n## ", 1)[0]

    found: set[tuple[str, str]] = set()
    for verbs, path in re.findall(r"`([A-Z|]+)\s+(/api/[^`\s]+)`", table):
        for verb in verbs.split("|"):
            found.add((verb, path))
    assert found, "the console API table listed no endpoints"
    return found


def _live_routes() -> set[tuple[str, str]]:
    """Every route the application actually serves."""
    live: set[tuple[str, str]] = set()
    for rule in flask_app.app.url_map.iter_rules():
        for method in rule.methods - IMPLICIT_METHODS:
            live.add((method, str(rule.rule)))
    return live


# --------------------------------------------------------- what exists
def test_the_documented_api_is_exactly_the_api_that_is_served():
    """The list a FastAPI port has to reproduce, and the list a cleanup has
    to justify. Drift in either direction is a failure: an undocumented
    endpoint is one nobody decided to support, and a documented one that is
    gone is a promise the product no longer keeps."""
    served = {row for row in _live_routes()
              if row not in PAGES | LEGACY_REDIRECTS | FRAMEWORK_ROUTES}
    documented = _documented_api()

    undocumented = sorted(served - documented)
    missing = sorted(documented - served)
    assert (undocumented, missing) == ([], []), (
        "console API drift.\n"
        "  served but not in the README table: " + repr(undocumented) + "\n"
        "  in the README table but not served: " + repr(missing)
    )


def test_every_route_is_either_api_page_or_a_declared_legacy_redirect():
    """Nothing is served that this file has not been told about. The point is
    the moment a route is added: it is classified then, by the person adding
    it, rather than archaeologically before a migration."""
    unclassified = sorted(
        _live_routes() - _documented_api() - PAGES - LEGACY_REDIRECTS - FRAMEWORK_ROUTES
    )
    assert unclassified == [], (
        "these routes belong to no declared group: " + repr(unclassified)
    )


def test_the_legacy_redirects_still_redirect_rather_than_render(client):
    """They carry no content of their own; that is why they are cleanup
    candidates and why removing them is a decision about bookmarks, not
    about behaviour."""
    for _, path in sorted(LEGACY_REDIRECTS):
        response = client.get(path)
        assert response.status_code in (301, 302, 308), path
        assert response.headers["Location"].endswith(("/", "/lab")), path


def test_the_two_routes_that_can_refuse_under_load_are_the_documented_two():
    """``/api/documents/upload`` and ``/api/query`` are the bounded pair, and
    the README says so. Every other endpoint is expected to answer."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "`POST /api/documents/upload` and `POST /api/query` are the two" in readme
    assert ("POST", "/api/documents/upload") in _documented_api()
    assert ("POST", "/api/query") in _documented_api()


# ---------------------------------------------------- what a refusal means
@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    with flask_app.app.test_client() as made:
        yield made


def _json(response):
    body = response.get_json()
    assert isinstance(body, dict), response.data[:200]
    return body


def test_a_malformed_request_is_a_400_and_reaches_nothing(client):
    """No question is not a server fault and not a missing resource. It also
    must not consume a query slot -- which is why it is answered before
    admission, and why a rewrite that validates after admission would be a
    regression this catches (``tests/integration/test_query_api.py`` holds
    the slot half)."""
    response = client.post("/api/query", json={"question": "   "})
    assert response.status_code == 400
    assert "error" in _json(response)


def test_an_unknown_resource_is_a_404_and_says_which_kind(client):
    """Four different resources, one code, and a body that names the kind.
    A rewrite that turns any of these into 500 loses the client's ability to
    tell "you asked for something that is not here" from "we broke"."""
    for path in ("/api/kb/kb-that-does-not-exist",
                 "/api/ingest/jobs/job-that-does-not-exist",
                 "/api/demo/viewer-analysis/doc-that-does-not-exist/payload"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert "error" in _json(response), path

    deleted = client.delete("/api/goldset/entry-that-does-not-exist")
    assert deleted.status_code == 404
    assert "error" in _json(deleted)


#: A rejected payload is a 400 that creates nothing. Not repeated here: it is
#: already driven at the level it belongs to, over the real managers --
#: ``tests/integration/test_kb_and_goldset_api.py`` (an invalid chunker is a
#: client error and creates nothing) and ``tests/unit/test_kb_lifecycle.py``
#: (a rejected payload leaves no record behind, in memory or on disk).


def test_a_health_check_answers_without_a_model_a_store_or_a_provider(client):
    """The container's healthcheck and the Viewer's console probe both call
    this, so it has to stay cheap and always answerable -- including while
    the service is degraded, which it reports rather than fails on."""
    response = client.get("/api/health")
    assert response.status_code in (200, 503)
    body = _json(response)
    assert "status" in body


#: What each refusal means to a client. The failure mode of a rewrite is not
#: losing one of these -- it is collapsing several into 500 or into 400, and
#: then every caller has to guess.
REFUSALS = {
    400: "the request or its payload is wrong",
    404: "the resource is not here",
    409: "the resource is here and in use",
    500: "the server failed",
    503: "capacity, or a capability, is unavailable now",
    504: "the deadline passed",
}


def test_the_product_still_makes_every_distinction_in_its_refusal_taxonomy():
    """Each of these codes is a decision some route reaches deliberately.

    ``409`` (a store another knowledge base still uses), ``503`` (overload,
    or an unavailable answer model) and ``504`` (a passed deadline) are
    *caused* by other suites, which is where they belong -- reaching them
    here would mean faking the condition rather than provoking it. What is
    checked here is that they have not silently left the codebase, which is
    what a framework port collapsing its error handling looks like from the
    outside.
    """
    import ast

    source = ast.parse((REPO / "app.py").read_text(encoding="utf-8"))
    returned = set()
    for node in ast.walk(source):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            for element in node.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, int):
                    returned.add(element.value)

    lost = sorted(code for code in REFUSALS if code not in returned)
    assert lost == [], (
        "no route answers with these any more, so the distinction they made "
        "is gone: " + repr({code: REFUSALS[code] for code in lost})
    )
