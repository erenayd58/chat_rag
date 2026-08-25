"""Which retrieval methods a configured retriever can actually serve.

The experiment screen used to offer Hybrid / Vector / BM25 unconditionally. On
the ``bm25_only`` profile that is wrong twice over: Vector raises, because the
profile loads no embedding model, and Hybrid is the very same code path as
BM25, because the profile's "hybrid" retriever *is* the lexical one. The user
saw a raw exception for one option and two identical options for the rest.

This module only *reports* what the retriever in hand can do. It runs no
search, changes no scoring and touches no retriever class: capability is
derived from the retriever's own declared contract
(``requires_document_embeddings``) plus which search entry points it exposes.
"""

from __future__ import annotations

from typing import Any, Dict, List

#: Order the UI should present them in.
METHODS = ("hybrid", "vector", "bm25")

LABELS = {
    "hybrid": "Hybrid",
    "vector": "Vector",
    "bm25": "BM25",
}


def _has(retriever: Any, name: str) -> bool:
    return callable(getattr(retriever, name, None))


def retrieval_capabilities(retriever: Any) -> Dict[str, Any]:
    """Report the methods ``retriever`` can serve, with a reason for each gap.

    Returns ``{"methods": [{name, label, available, reason}], "default": str,
    "dense": bool}``. ``default`` is the first available method, so a caller
    can preselect something that works.
    """
    dense = bool(getattr(retriever, "requires_document_embeddings", True))

    available: Dict[str, bool] = {}
    reasons: Dict[str, str] = {}

    # Dense retrieval needs both an entry point and a retriever that actually
    # keeps document vectors. BM25OnlyRetriever exposes vector_search only to
    # fail loudly, so the declared contract is what decides.
    if not _has(retriever, "vector_search"):
        available["vector"], reasons["vector"] = False, "Bu retriever vektör araması sunmuyor."
    elif not dense:
        available["vector"], reasons["vector"] = (
            False,
            "Bu profilde embedding hesaplanmıyor; dense arama yapılamaz.",
        )
    else:
        available["vector"] = True

    if _has(retriever, "keyword_search"):
        available["bm25"] = True
    else:
        available["bm25"], reasons["bm25"] = False, "Bu retriever ayrı BM25 araması sunmuyor."

    # Without a dense leg, "hybrid" is the lexical search under another name.
    # Offering both would suggest a comparison the user cannot actually make.
    if not _has(retriever, "hybrid_search"):
        available["hybrid"], reasons["hybrid"] = False, "Bu retriever hibrit arama sunmuyor."
    elif not dense and available.get("bm25"):
        available["hybrid"], reasons["hybrid"] = (
            False,
            "Dense bacak yok; hibrit arama BM25 ile aynı sonucu verir.",
        )
    else:
        available["hybrid"] = True

    methods: List[Dict[str, Any]] = []
    for name in METHODS:
        entry: Dict[str, Any] = {
            "name": name,
            "label": LABELS[name],
            "available": bool(available.get(name)),
        }
        if not entry["available"] and name in reasons:
            entry["reason"] = reasons[name]
        methods.append(entry)

    default = next((m["name"] for m in methods if m["available"]), None)
    return {"methods": methods, "default": default, "dense": dense}


def method_is_available(retriever: Any, method: str) -> bool:
    for entry in retrieval_capabilities(retriever)["methods"]:
        if entry["name"] == method:
            return bool(entry["available"])
    return False


def unavailable_reason(retriever: Any, method: str) -> str:
    for entry in retrieval_capabilities(retriever)["methods"]:
        if entry["name"] == method:
            return entry.get("reason") or f"'{method}' bu yapılandırmada kullanılamıyor."
    return f"Bilinmeyen retrieval yöntemi: '{method}'."
