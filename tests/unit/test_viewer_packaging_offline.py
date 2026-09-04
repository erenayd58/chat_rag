"""What the Viewer packaging worker is allowed to reach for.

The intent has always been that packaging costs no second provider call: an
ingest already paid for the expensive work, and the packager reuses it. The
existing tests hold that up from the *inside*, by asserting the call counts the
Deep pipeline records. These hold it up from the outside, which is a different
and stronger claim: with the network physically unavailable to this process,
packaging still completes.

The distinction matters for more than tidiness. Phase 2 has to size a global
provider limit, and the answer depends on whether the packaging thread is one
more consumer of provider concurrency or none. It is none -- and this file is
the evidence, so the number stays checkable rather than remembered.

The one thing packaging *does* reach for is recorded here too, because it is
easy to over-generalise from "no provider call" to "no model": the Hybrid
variant loads a local sentence-transformers boundary model. That is CPU and
memory on this machine, not a request to anyone, and on a machine with no model
cache it is a download. Standard, Markdown and Deep touch none of it.
"""

from __future__ import annotations

import socket

import pytest

from components.viewer import analysis
from components.viewer import methods as M

# The corpus builder, the Deep run factory and the analysis-root fixture that
# file already owns. Imported rather than copied so both files package the same
# document; pytest puts this directory on the path.
from test_viewer_analysis import _corpus, _deep_run, workspace  # noqa: F401


@pytest.fixture
def no_network(monkeypatch):
    """Make every outbound connection impossible, at the socket layer.

    Not a stub of one provider transport: anything that tried to open a
    connection -- an HTTP client, a model hub, a telemetry callback nobody
    remembered -- fails here. Loopback is refused too, so this cannot pass by
    talking to something on this machine.
    """
    opened: list[tuple] = []

    class RefusedSocket(socket.socket):
        def connect(self, address, *args, **kwargs):
            opened.append(address)
            raise AssertionError(f"packaging opened a network connection to {address!r}")

        def connect_ex(self, address, *args, **kwargs):
            opened.append(address)
            raise AssertionError(f"packaging opened a network connection to {address!r}")

    monkeypatch.setattr(socket, "socket", RefusedSocket)

    def refuse_create(address, *args, **kwargs):
        opened.append(address)
        raise AssertionError(f"packaging opened a network connection to {address!r}")

    monkeypatch.setattr(socket, "create_connection", refuse_create)
    return opened


def test_a_standard_upload_packages_with_no_network_at_all(workspace, no_network):
    """The default upload path: Standard asked for, Deep produced deterministically."""
    analysis.stage(doc_id="probe-doc", label="Probe belgesi", units=_corpus(),
                   kb_id="kb1", kb_name="probe-kb", chunking_mode="standard")
    analysis._queue.join()

    state = analysis.read_state("probe-doc")
    assert state["status"] == analysis.STATUS_READY, state.get("error")
    assert state["deep_source"] == analysis.SOURCE_DETERMINISTIC
    assert no_network == [], f"packaging reached out to {no_network}"


def test_a_deep_upload_packages_with_no_network_at_all(workspace, no_network):
    """A model-backed ingest hands its run over; packaging reuses it as it is."""
    units = _corpus()
    run = _deep_run(units)

    analysis.stage(doc_id="probe-doc", label="Probe belgesi", units=units, deep_result=run,
                   kb_id="kb1", kb_name="probe-kb", chunking_mode="deep_analysis")
    analysis._queue.join()

    state = analysis.read_state("probe-doc")
    assert state["status"] == analysis.STATUS_READY, state.get("error")
    assert state["deep_source"] == analysis.SOURCE_INGEST
    assert no_network == []


def test_the_packaged_deep_variant_records_no_provider_calls(workspace, no_network):
    analysis.stage(doc_id="probe-doc", label="Probe belgesi", units=_corpus(),
                   kb_id="kb1", kb_name="probe-kb", chunking_mode="standard")
    analysis._queue.join()

    deep = analysis.payload("probe-doc")["meta"]["deep"]
    assert deep["calls"] == {"proposer": 0, "verifier": 0, "total": 0}
    assert deep["estCostUsd"] == 0


# ------------------------------------------------- the one local model there is


def test_only_the_hybrid_variant_loads_a_model(workspace, monkeypatch):
    """"No provider call" is not "no model", and the difference is Hybrid's.

    Hybrid picks its cut points with a local sentence-transformers boundary
    model -- the same one the frozen benchmark's Hybrid arm used, so a live
    Hybrid variant is that arm rather than a lookalike. It is CPU and memory
    here, never a request to a provider, and it is the only variant that wants
    a model at all. Loading one is refused rather than performed, so this test
    downloads nothing.
    """
    from amsc import embeddings
    from amsc.models import RawDocumentUnit

    class ModelWanted(RuntimeError):
        pass

    def refuse(*args, **kwargs):
        raise ModelWanted("a boundary model was requested")

    monkeypatch.setattr(
        embeddings.SentenceTransformerBoundaryEmbedder, "from_pretrained", refuse
    )
    # ``_chunk_rows`` is handed the canonical the worker loaded from disk,
    # which is models rather than the raw rows ``stage`` was given.
    units = [RawDocumentUnit.model_validate(u) for u in _corpus()]

    for method in (M.STANDARD, M.MARKDOWN):
        rows = analysis._chunk_rows(method, units)
        assert rows, f"{method} produced no chunks"

    with pytest.raises(ModelWanted):
        analysis._chunk_rows(M.HYBRID, units)


def test_packaging_holds_no_provider_slot(workspace, no_network, monkeypatch):
    """The provider budget is Deep ingest's alone. Packaging a Deep upload,
    a Standard one and their variants leaves the budget's counters at zero,
    which is the number Phase 2 sized the limit against."""
    from components.chunker import deep_analysis
    from components.ingest import limits as L

    budget = L.configure_budget(2)

    def never(settings):  # pragma: no cover - the assertion is that it is not called
        raise AssertionError("packaging tried to build a provider transport")

    monkeypatch.setattr(deep_analysis, "build_transports", never)
    try:
        units = _corpus()
        analysis.stage(doc_id="probe-doc", label="Probe", units=units, kb_id="kb1", kb_name="probe-kb",
                       chunking_mode="deep_analysis", deep_result=_deep_run(units),
                       methods=[M.STANDARD, M.DEEP, M.MARKDOWN])
        analysis._queue.join()
        state = analysis.read_state("probe-doc")
        assert state["status"] == analysis.STATUS_READY, state
        assert budget.snapshot() == {"limit": 2, "inflight": 0, "peak": 0,
                                     "acquired_total": 0, "refused_total": 0}
    finally:
        L.configure_budget(8)
