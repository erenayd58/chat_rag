"""The HTTP surface, and what its refusals mean.

This module was written when the console was Flask and was going to stop
being Flask. A framework port is proven by parity, and parity needs a list of
what there is to be at parity with -- which nothing in this repository had.
Ninety tests drove individual routes; none said how many routes there are, so
a route lost in a port would be found only if some unrelated test happened to
use it, and a route quietly added would never be found at all.

The port is done and the Flask-era surface is gone (``docs/legacy-removal.md``),
so what this module pins is the surface that is left, in two halves.

**What exists.** The endpoint tables in ``docs/api-v1.md`` are the contract and
are held against the application's real routing table in both directions: a
new endpoint is documented in the same commit that adds it, and a removal is a
removal from both. Exactly one route is served that the contract does not
publish -- ``GET /api/ops/metrics``, the operator surface -- and it is declared
here and published in ``docs/operations.md``, because a route no document
claims is a route nobody decided to support.

**What a refusal means.** The distinctions the product makes are a contract: a
malformed request, an unknown resource, a rejected payload and a server fault
are four different answers, and a client branches on them. They are asserted
here as a taxonomy rather than one at a time, because the failure mode of a
rewrite is not losing one code -- it is collapsing several into 500 or into
400.

What this module does **not** pin: any error *message*, the internals of the
routing table, the framework's own behaviour (its automatic ``HEAD``, its
405), or the JSON body of a success. Those belong to the implementation and
may all change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http

REPO = Path(__file__).resolve().parents[2]
V1 = http.v1.PREFIX

#: The one served route that is not on the contract. It is an operator
#: surface: its *contents* are deliberately free to change with the internals
#: they report on, which is why it was never promoted to a versioned path, and
#: it kept that path and that body through the removal of the Flask surface
#: that used to serve it. Published in ``docs/operations.md``.
OPERATOR_ROUTES = {("GET", "/api/ops/metrics")}


def _endpoints(text: str) -> set[tuple[str, str]]:
    """Every ``VERB /api/...`` a piece of documentation publishes."""
    found: set[tuple[str, str]] = set()
    for verbs, path in re.findall(r"`([A-Z|]+)\s+(/api/[^`\s]+)`", text):
        for verb in verbs.split("|"):
            found.add((verb, path))
    return found


def _documented_v1() -> set[tuple[str, str]]:
    """The endpoints ``docs/api-v1.md`` publishes as the product contract.

    A document that *is* the contract has to be the whole list rather than a
    sample of it, so it is held against the routing table in both directions.
    """
    doc = REPO / "docs" / "api-v1.md"
    assert doc.exists(), "the /api/v1 contract has no published endpoint list"
    found = _endpoints(doc.read_text(encoding="utf-8"))
    assert found, "docs/api-v1.md listed no endpoints"
    return found


def _live_routes() -> set[tuple[str, str]]:
    """Every route the application actually serves, in the documents' spelling."""
    return http.surface(entrypoint.application)


# --------------------------------------------------------- what exists
def test_the_documented_api_is_exactly_the_api_that_is_served():
    """Drift in either direction is a failure: an undocumented endpoint is one
    nobody decided to support, and a documented one that is gone is a promise
    the product no longer keeps."""
    served = _live_routes()
    documented = _documented_v1() | OPERATOR_ROUTES

    undocumented = sorted(served - documented)
    missing = sorted(documented - served)
    assert (undocumented, missing) == ([], []), (
        "API drift.\n"
        "  served but published nowhere: " + repr(undocumented) + "\n"
        "  published but not served: " + repr(missing)
    )


def test_the_product_contract_and_the_operator_surface_stay_apart():
    """``/api/v1`` is the contract; the operator route deliberately is not.

    Each served route belongs to exactly one of them and is published in that
    one's own place, so "is this supported long term?" is answered by the path
    rather than by asking somebody. It is also what stops the two drifting
    into each other: an operator endpoint quietly added under ``/api/v1``, or
    a contract endpoint served outside it, fails here.
    """
    served = _live_routes()
    versioned = {row for row in served if row[1].startswith(f"{V1}/")}
    assert versioned, "the product contract serves nothing"

    assert versioned == _documented_v1(), (
        "docs/api-v1.md and the served /api/v1 disagree: "
        + repr(sorted(versioned ^ _documented_v1()))
    )
    assert (served - versioned) == OPERATOR_ROUTES, (
        "something outside /api/v1 is served that this file has not been told "
        "about: " + repr(sorted((served - versioned) ^ OPERATOR_ROUTES))
    )


def test_the_operator_surface_is_published_where_an_operator_would_look():
    """It is off the contract, which is a reason for it not to be versioned --
    not a reason for it to be undocumented."""
    operations = (REPO / "docs" / "operations.md").read_text(encoding="utf-8")
    for _, path in sorted(OPERATOR_ROUTES):
        assert path in operations, f"{path} is served and docs/operations.md does not mention it"


def test_the_operator_surface_is_not_on_the_generated_contract(client):
    """``/api/v1/openapi.json`` is the list a client may build against, and
    this route is not on it -- deliberately, because its body is free to
    change with what it reports on."""
    document = client.get(http.v1.OPENAPI_PATH).json()
    for _, path in sorted(OPERATOR_ROUTES):
        assert path not in document["paths"], f"{path} was promoted by accident"


def test_the_operator_surface_answers_where_it_always_did(client):
    """The path and the body did not move when the surface under them was
    removed: a script polling it sees no difference."""
    response = client.get("/api/ops/metrics")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    for key in ("state", "ready", "reasons", "ingest", "query", "metrics",
                "database", "caches", "configuration"):
        assert key in body, key


def test_the_one_route_that_can_refuse_under_load_is_the_documented_one():
    """``POST /api/v1/queries`` is the bounded one, and the README says so.
    The other half of the old pair -- a synchronous upload -- went with the
    surface that offered it: ``POST /api/v1/documents`` always answers 202."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "`POST /api/v1/queries` is the one that can refuse" in readme
    assert ("POST", f"{V1}/queries") in _documented_v1()
    assert ("POST", f"{V1}/documents") in _documented_v1()


# ---------------------------------------------------- what a refusal means
@pytest.fixture
def client():
    with TestClient(http.create_app(entrypoint.services),
                    raise_server_exceptions=False) as made:
        yield made


def _error(response) -> dict:
    body = response.json()
    assert isinstance(body, dict) and "error" in body, response.text
    return body["error"]


def test_a_malformed_request_is_a_400_and_reaches_nothing(client):
    """No question is not a server fault and not a missing resource. It also
    must not consume a query slot -- which is why it is answered before
    admission, and why a rewrite that validates after admission would be a
    regression this catches (``tests/integration/test_query_api.py`` holds
    the slot half)."""
    response = client.post(f"{V1}/queries", json={"question": "   "})
    assert response.status_code == 400
    assert _error(response)["type"] == "invalid_request"


def test_an_unknown_resource_is_a_404_and_says_which_kind(client):
    """Three different resources, one code, and a body that names the kind.
    A rewrite that turns any of these into 500 loses the client's ability to
    tell "you asked for something that is not here" from "we broke"."""
    for path in (f"{V1}/knowledge-bases/kb-that-does-not-exist",
                 f"{V1}/ingest-jobs/job-that-does-not-exist",
                 f"{V1}/documents/doc-that-does-not-exist"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert _error(response)["type"] == "not_found", path


def test_a_resource_that_is_here_but_not_built_yet_is_a_409(client):
    """The distinction a client polling for a build needs: "not here" and "not
    ready" are different answers, and the second carries the state."""
    response = client.get(f"{V1}/documents/doc-that-does-not-exist/analysis/payload")
    assert response.status_code == 409
    error = _error(response)
    assert error["type"] == "not_ready"
    assert "state" in error["details"]


#: A rejected payload is a 400 that creates nothing. Not repeated here: it is
#: already driven at the level it belongs to, over the real managers --
#: ``tests/integration/test_kb_api.py`` (an invalid chunker is a
#: client error and creates nothing) and ``tests/unit/test_kb_lifecycle.py``
#: (a rejected payload leaves no record behind, in memory or on disk).


def test_a_health_check_answers_without_a_model_a_store_or_a_provider(client):
    """The container's healthcheck and the demo launcher both call this, so it
    has to stay cheap and always answerable -- including while the service is
    degraded, which it reports rather than fails on."""
    response = client.get(f"{V1}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] in {"ok", "overloaded", "degraded"}
    assert body["ready"] is True


#: What each refusal means to a client. The failure mode of a rewrite is not
#: losing one of these -- it is collapsing several into 500 or into 400, and
#: then every caller has to guess.
REFUSALS = {
    400: "the request or its payload is wrong",
    404: "the resource is not here",
    409: "the resource is here and in use, or not built yet",
    500: "the server failed",
    503: "capacity, or a capability, is unavailable now",
    504: "the deadline passed",
}


def test_the_product_still_makes_every_distinction_in_its_refusal_taxonomy():
    """Each of these codes is a decision the adapter reaches deliberately.

    ``503`` (overload, or an unavailable answer model) and ``504`` (a passed
    deadline) are *caused* by other suites, which is where they belong --
    reaching them here would mean faking the condition rather than provoking
    it. What is checked here is that they have not silently left the codebase,
    which is what a framework port collapsing its error handling looks like
    from the outside.

    The codes live in the HTTP adapter and nowhere else: one table
    (``v1.errors.REFUSALS``, which maps an application refusal to a status and
    a name) plus the few literals the routers declare on their decorators. A
    port replaces this package and has to reproduce the same six distinctions
    in whatever it writes instead.
    """
    import ast

    adapter = REPO / "interfaces" / "http"
    assert adapter.is_dir(), "the HTTP adapter package is gone"

    returned = set()
    for path in sorted(adapter.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            # A status returned beside a body ...
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                returned |= {e.value for e in node.value.elts
                             if isinstance(e, ast.Constant) and isinstance(e.value, int)}
            # ... or one declared in a translation table.
            elif isinstance(node, ast.Dict):
                for value in node.values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, int):
                        returned.add(value.value)
                    elif isinstance(value, ast.Tuple) and value.elts:
                        first = value.elts[0]
                        if isinstance(first, ast.Constant) and isinstance(first.value, int):
                            returned.add(first.value)
            # ... or one a handler builds directly.
            elif isinstance(node, ast.Call):
                for argument in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, int):
                        returned.add(argument.value)

    lost = sorted(code for code in REFUSALS if code not in returned)
    assert lost == [], (
        "no adapter answers with these any more, so the distinction they made "
        "is gone: " + repr({code: REFUSALS[code] for code in lost})
    )
