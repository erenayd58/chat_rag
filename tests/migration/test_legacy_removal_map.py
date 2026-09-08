"""The removal map is held against the routing table, not just written down.

``docs/legacy-removal.md`` is the plan Steps 11 to 13 work from: every endpoint
the Flask-era surface still serves, who calls it, what answers it on the
contract, and which wave removes it. A plan written before the work is only
worth anything if it cannot go stale, so it is compared with the application's
real routing table in both directions -- exactly as ``test_http_surface.py``
compares the README's console-API table and ``docs/api-v1.md``.

Three failures this catches, all of which have happened to plans like it:

* a legacy endpoint added and not classified, so nobody decided when it goes;
* a legacy endpoint removed and left on the page, so the plan promises work
  that is done;
* a replacement named on the page that `/api/v1` does not actually serve --
  the one that turns into a client rewrite discovered in the wrong week.

What is deliberately *not* checked here is the caller column. It is evidence
gathered by reading two repositories and a container file; a test that
re-derived it would be re-deriving it wrongly, and the routing table cannot
confirm who calls what.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import app as flask_app
from interfaces.http import v1

REPO = Path(__file__).resolve().parents[2]
MAP = REPO / "docs" / "legacy-removal.md"

#: Rows of the removal map: ``| VERB /path | replacement | caller |``.
ROW = re.compile(r"^\|\s*`([A-Z]+)\s+(/api/[^`]+)`\s*\|\s*(.*?)\s*\|", re.M)
#: A replacement cell either names one `/api/v1` endpoint or is an em dash.
REPLACEMENT = re.compile(r"`([A-Z]+)\s+(/api/v1/[^`]+)`")

#: Flask adds these itself; a plan does not classify them.
IMPLICIT = {"HEAD", "OPTIONS"}
#: Routes that are not the legacy API and are classified in prose instead:
#: the rendered screens, Flask's static endpoint, and the contract itself.
NOT_LEGACY_API = re.compile(r"^(/api/v1/|/static/|/$|/chat$|/lab$|/kb/)")


def _mapped() -> dict[tuple[str, str], str]:
    """Every legacy endpoint the map classifies, and what it says replaces it."""
    text = MAP.read_text(encoding="utf-8")
    rows = {(verb, path): replacement for verb, path, replacement in ROW.findall(text)}
    assert rows, "the removal map classifies no endpoint"
    return rows


def _served_legacy() -> set[tuple[str, str]]:
    """Every legacy API route the application actually serves."""
    live = set()
    for rule in flask_app.app.url_map.iter_rules():
        path = str(rule.rule)
        if NOT_LEGACY_API.match(path):
            continue
        for method in rule.methods - IMPLICIT:
            live.add((method, path))
    return live


def _served_v1() -> set[tuple[str, str]]:
    """Every `/api/v1` route, in the map's spelling (``<id>``, not ``{id}``)."""
    live = set()
    for rule in flask_app.app.url_map.iter_rules():
        path = str(rule.rule)
        if not path.startswith(v1.PREFIX):
            continue
        for method in rule.methods - IMPLICIT:
            live.add((method, path))
    return live


def test_every_legacy_endpoint_served_is_classified_exactly_once():
    """Both directions. An unclassified endpoint is one nobody decided about;
    a classified one that is gone is a plan promising finished work."""
    assert _mapped().keys() == _served_legacy(), sorted(
        set(_mapped()) ^ _served_legacy())


def test_no_endpoint_is_classified_into_two_waves():
    """A route in two tables would be removed twice, or once and then missed."""
    text = MAP.read_text(encoding="utf-8")
    seen = [(verb, path) for verb, path, _ in ROW.findall(text)]
    duplicates = sorted({row for row in seen if seen.count(row) > 1})
    assert duplicates == [], duplicates


def test_every_replacement_named_is_a_route_the_contract_serves():
    """The column a Step 11 client is written from. A replacement that does not
    exist is a rewrite discovered after the screen is built."""
    served = _served_v1()
    missing = []
    for (verb, path), replacement in _mapped().items():
        for answer in REPLACEMENT.findall(replacement):
            if answer not in served:
                missing.append(f"{verb} {path} -> {answer[0]} {answer[1]}")
    assert missing == [], missing


def test_an_endpoint_with_no_replacement_says_so_rather_than_leaving_the_cell_blank():
    """`—` is a decision to stop offering something, and the reason is in
    ``docs/api-v1.md``. An empty cell is an unanswered question."""
    unexplained = [f"{verb} {path}" for (verb, path), replacement in _mapped().items()
                   if not replacement.strip()]
    assert unexplained == [], unexplained


@pytest.mark.parametrize("wave", ["## Wave 1", "## Wave 2", "## Wave 3"])
def test_all_three_waves_are_present(wave):
    """The order is the plan. A missing wave is a plan that removes everything
    at once, which is what this page exists to prevent."""
    assert wave in MAP.read_text(encoding="utf-8")
