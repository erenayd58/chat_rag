"""The companion "Agentic Chunking Viewer" link and its status probe.

The console links to the chunk repository's Viewer v2 server. The address is
configuration (VIEWER_URL; empty hides the link), the sidebar shows whether
the viewer answers, and the probe is a backend call so the browser never has
to reach a second origin itself.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

import app as flask_app


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    with flask_app.app.test_client() as test_client:
        yield test_client


def test_the_viewer_address_is_configuration(monkeypatch):
    from config.settings import Settings

    monkeypatch.delenv("VIEWER_URL", raising=False)
    assert Settings().viewer_url == "http://127.0.0.1:8765/"
    monkeypatch.setenv("VIEWER_URL", "http://demo-host:9000/")
    assert Settings().viewer_url == "http://demo-host:9000/"
    monkeypatch.setenv("VIEWER_URL", "")
    assert Settings().viewer_url == ""


def test_the_sidebar_links_to_the_configured_viewer(client, monkeypatch):
    monkeypatch.setattr(flask_app.settings, "viewer_url", "http://127.0.0.1:8765/")
    page = client.get("/lab").get_data(as_text=True)
    assert 'id="viewerLink"' in page
    assert 'href="http://127.0.0.1:8765/"' in page
    assert 'target="_blank"' in page and 'rel="noopener"' in page
    assert "Chunking Viewer" in page
    assert 'id="viewerCardState"' in page  # the Lab card


def test_an_empty_address_hides_the_link(client, monkeypatch):
    monkeypatch.setattr(flask_app.settings, "viewer_url", "")
    page = client.get("/lab").get_data(as_text=True)
    assert 'id="viewerLink"' not in page
    assert "Open Viewer" not in page


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_the_probe_reports_a_running_viewer(client, monkeypatch):
    import urllib.request

    seen = {}

    def fake_urlopen(url, timeout):
        seen["url"] = url
        seen["timeout"] = timeout
        return _Response(json.dumps({"documents": ["kkb-2024", "kkb-2022"], "arms": []}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(flask_app.settings, "viewer_url", "http://127.0.0.1:8765/")
    body = client.get("/api/demo/viewer").get_json()
    assert body == {
        "success": True, "configured": True, "url": "http://127.0.0.1:8765/",
        "reachable": True, "documents": 2,
    }
    assert seen["url"] == "http://127.0.0.1:8765/api/health"
    assert seen["timeout"] <= 2, "a status dot must not stall the page"


def test_the_probe_reports_a_stopped_viewer_without_raising(client, monkeypatch):
    import urllib.error
    import urllib.request

    def fake_urlopen(url, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(flask_app.settings, "viewer_url", "http://127.0.0.1:8765/")
    body = client.get("/api/demo/viewer").get_json()
    assert body["reachable"] is False and body["configured"] is True


def test_an_unconfigured_viewer_is_not_probed(client, monkeypatch):
    import urllib.request

    def fake_urlopen(url, timeout):  # pragma: no cover - must not be called
        raise AssertionError("no probe without an address")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(flask_app.settings, "viewer_url", "")
    body = client.get("/api/demo/viewer").get_json()
    assert body["configured"] is False and body["reachable"] is False
