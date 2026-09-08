"""What this deployment can actually offer, as opposed to what it knows about.

Three questions, one shape of answer: which chunking methods this machine can
run, which models this knowledge base's pipeline is wired to, and which
retrieval methods its retriever can serve. All three are read-only, and all
three exist so a screen never offers something the backend would then refuse
-- an unavailable option carries its reason instead of quietly disappearing.
"""

from __future__ import annotations

from typing import Optional

from components.retriever import retrieval_capabilities
from components.viewer import methods as viewer_methods


def chunking_methods() -> list[dict]:
    """The chunking methods this deployment can run.

    The upload form and the Viewer both read this, so neither can offer a
    method the machine cannot produce. The shape is frozen: it is the body of
    a compatibility endpoint, and ``tests/unit/test_method_wire_contract.py``
    holds it verbatim.
    """
    return viewer_methods.catalogue()


def chunking_method_facts() -> list[dict]:
    """The same methods, with the rest of what the registry knows about them.

    Two functions rather than one because the first is a frozen wire shape and
    this is not: a picker wants to know that a method is an *orchestration*
    over a baseline partition rather than a partition, and that a method needs
    a local embedding model, and neither belongs in a body that predates the
    question. Both read the one registry.
    """
    facts = []
    for entry in viewer_methods.catalogue():
        method = viewer_methods.METHODS[entry["key"]]
        facts.append({
            **entry,
            "needs_embedder": method.needs_embedder,
            "orchestration": method.deep,
            "baseline": method.baseline,
        })
    return facts


def model_chain(services, *, session_id: str, kb_id: Optional[str] = None) -> dict:
    """The configured model chain (agentic chunking, embedding, answer).

    Names, ids and endpoints only -- never a key.
    """
    return services.get_pipeline(session_id, kb_id).model_chain()


def retrieval(services, *, session_id: str, kb_id: Optional[str] = None) -> dict:
    """Which retrieval methods the configured retriever can actually serve.

    Reporting only: no search runs and no scoring changes. The review screen
    uses this to stop offering Vector on a profile that computes no embeddings,
    and to stop offering Hybrid where it is BM25 under another name.
    """
    return retrieval_capabilities(services.get_pipeline(session_id, kb_id).hybrid_retriever)
