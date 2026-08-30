"""The chunking methods a document can be analysed with, in one registry.

One upload, one parse, one canonical -- then as many chunkers as the user
asked for, each run over that same canonical and packaged in the shape the
Viewer reads. A method is listed here only if this deployment can actually
run it end to end; anything that would need a component we do not have is
declared unavailable *with its reason*, so a screen never offers a comparison
it cannot produce and never quietly drops one it was asked for.

The token budget is shared by every method on purpose: the comparison is
about where the boundaries fall, so letting one arm run to a different budget
would compare two things at once.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

#: Viewer arm id -> method. The ids are the Viewer's own arm names, so a
#: packaged variant needs no translation on the way in.
STANDARD = "structure-only"
DEEP = "agentic"
MARKDOWN = "markdown"
HYBRID = "hybrid"


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


#: Every method the product knows about, offered or not.
METHODS: dict[str, Method] = {
    MARKDOWN: Method(
        key=MARKDOWN,
        label="Markdown",
        summary=(
            "Metni sabit boyutta keser, bölüm yapısına bakmaz. Hızlıdır ve "
            "karşılaştırmada taban çizgisi olarak durur."
        ),
        engine="markdown_recursive",
    ),
    STANDARD: Method(
        key=STANDARD,
        label="Standard",
        summary=(
            "Dokümanın kendi başlık yapısını takip eder: her bölüm kendi başlığı "
            "altında kalır, yalnız çok büyüyen bölümler bölünür."
        ),
        engine="structure_first",
    ),
    DEEP: Method(
        key=DEEP,
        label="Deep Analysis",
        summary=(
            "Standard'ın bıraktığı kötü sınırları arar ve düzeltir; kararsız "
            "kalınan yerlerde modele danışır, her öneriyi iki kez doğrulatır."
        ),
        engine="deep_analysis",
        uses_model=True,
    ),
    HYBRID: Method(
        key=HYBRID,
        label="Hybrid",
        summary=(
            "Yapıyı takip eder, ama bütçeyi aşan bölümlerde kesim yerini anlam "
            "benzerliğine bakarak seçer."
        ),
        engine="hybrid_h1",
        needs_embedder=True,
    ),
}


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

#: The order every screen lists methods in.
ORDER = (MARKDOWN, STANDARD, DEEP, HYBRID)

#: What an upload gets when it asks for nothing in particular.
DEFAULT_SELECTION = (STANDARD,)


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
    method = METHODS.get(key)
    return method.label if method else str(key)


def labels(keys: Sequence[str]) -> list[str]:
    return [label(key) for key in keys]
