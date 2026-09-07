"""The chunking methods a document can be analysed with, as the product offers them.

The registry is the library's: ``amsc.chunking.registry`` is where a method is added,
and it owns the method's identity -- wire key, engine kind, product name,
summary, capabilities -- because the library's own surfaces (the Viewer
builders, the packager, the benchmark) read it too. This module is the
console's *view* of that registry. It adds exactly what only this deployment
can decide:

* **availability** -- a method that needs a sentence-embedding model is
  offered only when this machine can actually run it, and is declared
  unavailable *with its reason* otherwise, so a screen never offers a
  comparison it cannot produce and never quietly drops one it was asked for;
* **display order and the default** -- what the console lists first and
  what an upload gets when it asks for nothing in particular.

Everything else -- ``METHODS``, ``ORDER``, the key constants -- is read from
the registry as it is now, so a method registered there (a fifth one, in a
test) is known here without a second registration.

One upload, one parse, one canonical -- then as many chunkers as the user
asked for, each run over that same canonical and packaged in the shape the
Viewer reads. The token budget is shared by every method on purpose: the
comparison is about where the boundaries fall, so letting one arm run to a
different budget would compare two things at once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence as _Sequence
from dataclasses import dataclass, replace
from typing import Any, Iterator, Sequence

from amsc.chunking import registry

#: The Viewer arm ids of the four shipped methods. Read from the registry so
#: a rename there is a rename here; kept as names because every route and
#: test speaks in them.
STANDARD = registry.STANDARD.key
DEEP = registry.DEEP.key
MARKDOWN = registry.MARKDOWN.key
HYBRID = registry.HYBRID.key


@dataclass(frozen=True)
class Method:
    """One chunking method, as the product offers it."""

    key: str
    #: The name every screen uses. One name per method, everywhere.
    label: str
    #: One sentence: what this method does, for someone choosing it.
    summary: str
    #: The engine behind it, kept for the audit surfaces.
    engine: str
    #: False when this deployment cannot run it; ``reason`` says why.
    available: bool = True
    reason: str = ""
    #: Deep Analysis is the only method that may consult a model.
    uses_model: bool = False
    #: Hybrid is the only method that needs a sentence-embedding model.
    needs_embedder: bool = False
    #: An orchestration over a baseline partition rather than a partition.
    deep: bool = False
    #: For a deep method, the key of the partition it starts from.
    baseline: str | None = None


def _from_registry(entry: registry.ChunkMethod) -> Method:
    return Method(
        key=entry.key,
        label=entry.label,
        summary=entry.summary,
        engine=entry.kind,
        uses_model=entry.uses_model,
        needs_embedder=entry.needs_embedder,
        deep=entry.deep,
        baseline=entry.baseline,
    )


#: The order the console lists methods in. A product choice, distinct from
#: the library's own order (the Viewer page lists the benchmark arms first).
#: Any method registered but not named here is listed after these, so a new
#: method needs no edit to appear.
PRODUCT_ORDER: tuple[str, ...] = (MARKDOWN, STANDARD, DEEP, HYBRID)


def order() -> tuple[str, ...]:
    registered = registry.order()
    return tuple(key for key in PRODUCT_ORDER if key in registered) + tuple(
        key for key in registered if key not in PRODUCT_ORDER
    )


class _Order(_Sequence):
    """``ORDER`` as a live sequence: what the registry holds, in product order."""

    def __getitem__(self, index):
        return order()[index]

    def __len__(self) -> int:
        return len(order())

    def __iter__(self) -> Iterator[str]:
        return iter(order())

    def __eq__(self, other: object) -> bool:
        return tuple(order()) == tuple(other) if isinstance(other, _Sequence) else NotImplemented

    def __hash__(self) -> int:
        return hash(order())

    def __repr__(self) -> str:
        return repr(order())


class _Methods(Mapping):
    """``METHODS`` as a live mapping over the registry, in product order."""

    def __getitem__(self, key: str) -> Method:
        return _from_registry(registry.get(key))

    def __iter__(self) -> Iterator[str]:
        return iter(order())

    def __len__(self) -> int:
        return len(order())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and registry.is_known(key)

    def __repr__(self) -> str:
        return repr({key: self[key] for key in order()})


#: Every method the product knows about, offered or not.
METHODS: Mapping[str, Method] = _Methods()
#: The order every screen lists methods in.
ORDER: Sequence[str] = _Order()


#: The sentence-transformers model the frozen benchmark's Hybrid arm used. A
#: live Hybrid variant runs the same one, or it does not run at all.
BOUNDARY_MODEL = "intfloat/multilingual-e5-base"
_embedder_probe: tuple[bool, str] | None = None


def embedder_available() -> tuple[bool, str]:
    """Whether this machine can actually run the semantic boundary model.

    Probed once, without downloading: an arm we cannot produce must be
    offered as unavailable *with its reason*, never as a checkbox that
    quietly yields nothing.
    """
    global _embedder_probe
    if _embedder_probe is None:
        try:
            import sentence_transformers  # noqa: F401
        except Exception:  # noqa: BLE001
            _embedder_probe = (False, "Sınır modeli için gereken kütüphane kurulu değil.")
        else:
            from pathlib import Path as _Path
            import os as _os

            home = _os.environ.get("HF_HOME") or str(_Path.home() / ".cache" / "huggingface")
            cached = _Path(home) / "hub" / ("models--" + BOUNDARY_MODEL.replace("/", "--"))
            _embedder_probe = (
                (True, "")
                if cached.is_dir()
                else (False, f"{BOUNDARY_MODEL} modeli bu makinede indirilmemiş.")
            )
    return _embedder_probe


#: What an upload gets when it asks for nothing in particular.
DEFAULT_SELECTION = (STANDARD,)


def get(key: str) -> Method:
    """One method by key; ``KeyError`` (naming the known keys) for an unknown one."""
    return METHODS[key]


def resolve(key: str) -> Method:
    """One method with its availability decided against this machine."""
    method = METHODS[key]
    if method.needs_embedder:
        ok, why = embedder_available()
        if not ok:
            return replace(method, available=False, reason=why)
    return method


def offered() -> list[Method]:
    """The methods a user may pick, in display order."""
    return [m for m in (resolve(key) for key in ORDER) if m.available]


def catalogue() -> list[dict[str, Any]]:
    """Every method as the console's own API describes it."""
    return [
        {
            "key": m.key,
            "label": m.label,
            "summary": m.summary,
            "engine": m.engine,
            "available": m.available,
            "reason": m.reason,
            "uses_model": m.uses_model,
            # What an upload gets with nothing ticked; the form preselects it.
            "default": m.key in DEFAULT_SELECTION,
        }
        for m in (resolve(key) for key in ORDER)
    ]


def normalise(selection: Any) -> list[str]:
    """The methods actually to be run, from whatever the request sent.

    Unknown and unavailable names are dropped rather than half-honoured, and
    an empty selection falls back to Standard: a document with no analysis at
    all would be a document the Viewer cannot open.
    """
    if selection is None:
        raw: Sequence[str] = ()
    elif isinstance(selection, str):
        raw = [part.strip() for part in selection.replace(";", ",").split(",")]
    else:
        raw = [str(part).strip() for part in selection]
    chosen = [key for key in ORDER if key in raw and resolve(key).available]
    return chosen or list(DEFAULT_SELECTION)


def label(key: str) -> str:
    return METHODS[key].label if key in METHODS else str(key)


def labels(keys: Sequence[str]) -> list[str]:
    return [label(key) for key in keys]
