"""What ``chat_rag`` promises a consumer, held to as a list rather than a habit.

L4 published a facade. A published surface that nothing checks is a habit: the
next person to add a convenience method reaches for the container it is in
front of, returns whatever that call returned, and the promise quietly grows a
``Services`` in it. Nobody notices until a consumer's type checker does, or
until removing an internal class breaks somebody's code.

Three claims, and each is a different way to break the same promise:

**The list is the surface.** ``chat_rag.api.__all__`` is the published API, and
what the package actually exports has to be exactly it -- neither more (a name
that escaped by being importable) nor less (a name promised and not there).
``chat_rag`` re-exports the same list, so ``from chat_rag import Engine`` and
``from chat_rag.api import Engine`` cannot disagree.

**Internals stay inside.** Every public callable's signature -- parameters and
return -- may name only builtins, typing constructs, published types and the
configuration value objects (which are value objects, and are checked to still
be). An annotation naming ``Services``, ``RAGPipeline``, a repository or a
session -- anything out of :mod:`chat_rag.application`,
:mod:`chat_rag.components`, :mod:`chat_rag.pipeline`, :mod:`chat_rag.storage`,
:mod:`chat_rag.runtime`, :mod:`chat_rag.core` or :mod:`chat_rag.utils` -- is a
leak, and the one that is deliberate is declared below with its reason.

Returns and parameters are held to different standards on purpose. A return
may be an escape hatch: a caller can ignore it. A **parameter** may not -- a
method that demands a ``Services`` is a method only this repository can call,
and no declaration makes that acceptable.

**The refusals are part of it.** A caller has to be able to catch what this
library raises, so the exception classes are published too, and they are the
application's own -- not a second hierarchy wrapping them.

The check reads annotations rather than trusting docstrings, so a leak arrives
as a failure here rather than as a surprise in a consumer's editor.
"""

from __future__ import annotations

import inspect
import typing
from pathlib import Path

import pytest

import chat_rag
import chat_rag.api as api

REPO = Path(__file__).resolve().parents[2]

#: The escape hatches, declared. Each is a public member whose type is
#: deliberately an internal one, with the reason it is worth the exception.
#: Anything not on this list that names an internal type fails.
#:
#: They are all *reads*. Nothing published takes an internal type as an
#: argument, which is the direction that would matter most: a caller who
#: cannot construct what a method demands has no API at all.
DECLARED_ESCAPES = {
    "Engine.services":
        "the container underneath, so a caller can reach a use case this "
        "facade does not wrap. Documented as the escape hatch it is, and "
        "using it means taking on the session id and the activation.",
}

#: Packages whose names may not appear in a public signature.
INTERNAL_PACKAGES = (
    "chat_rag.application", "chat_rag.components", "chat_rag.pipeline",
    "chat_rag.storage", "chat_rag.runtime", "chat_rag.core", "chat_rag.utils",
    "chat_rag.process",
)

#: The one package that is neither published by name nor internal.
#:
#: L2 made configuration a set of **value objects a caller constructs**:
#: ``Settings`` is a plain dataclass whose fields are frozen dataclasses --
#: ``IngestLimits``, ``QueryLimits``, ``DatabaseSettings``, ``PathSettings``,
#: ``RuntimeLimits`` -- and ``EngineConfig`` is openly a stated subset of it.
#: Listing all six in ``__all__`` would bury the surface a caller actually
#: uses; pretending they are internal, while ``Settings`` is published and
#: takes them, would be a lie the guard tells on itself.
#:
#: The exemption is kept narrow by :func:`test_only_value_objects_reach_a_public_signature`
#: below: whatever comes out of here has to be a dataclass. A manager, a store
#: or a session appearing in a signature would fail exactly as before.
CONFIGURATION_PACKAGE = "chat_rag.config"


# ------------------------------------------------------------------ the list
def test_the_package_exports_exactly_what_it_publishes():
    """No name escapes by being importable, and none is promised and missing."""
    published = set(api.__all__)
    for name in published:
        assert hasattr(api, name), f"{name} is in __all__ and does not exist"

    # Everything else the module namespace holds must be a module, a private
    # name, or one of the submodules this package is made of.
    modules = {"config", "engine", "resources", "results"}
    # ``annotations`` is the __future__ feature object every module in this
    # package imports; it is not an export.
    escaped = {
        name for name in vars(api)
        if not name.startswith("_") and name not in published and name not in modules
        and name != "annotations" and not inspect.ismodule(getattr(api, name))
    }
    assert escaped == set(), f"exported but not published: {sorted(escaped)}"


def test_the_two_import_paths_publish_the_same_list():
    """``from chat_rag import Engine`` and ``from chat_rag.api import Engine``
    are the same promise, so one list cannot drift from the other."""
    assert set(chat_rag.__all__) == set(api.__all__)
    for name in chat_rag.__all__:
        assert getattr(chat_rag, name) is getattr(api, name), name


def test_every_published_name_is_a_class_or_a_function():
    """A published module would be a surface nobody could enumerate."""
    for name in api.__all__:
        member = getattr(api, name)
        assert inspect.isclass(member) or inspect.isfunction(member), (
            f"{name} is a {type(member).__name__}")


def test_the_refusals_are_published_and_are_the_applications_own():
    """A caller has to be able to catch what this library raises, and there is
    one hierarchy rather than a facade-shaped copy of it."""
    from chat_rag.application import errors
    from chat_rag.core import exceptions

    for name in ("ApplicationError", "InvalidRequest", "NotFound", "Conflict",
                 "Unavailable", "NotReady", "ProcessingFailed"):
        assert name in api.__all__, f"{name} is raised and not published"
        assert getattr(api, name) is getattr(errors, name)
    for name in ("IngestOverloaded", "IngestInterrupted", "QueryOverloaded",
                 "QueryTimeout"):
        assert name in api.__all__, f"{name} is raised and not published"
        assert getattr(api, name) is getattr(exceptions, name)


def test_the_version_is_the_distributions_own():
    """One place a release number lives, and it is not a second constant."""
    assert isinstance(chat_rag.__version__, str)
    assert chat_rag.__version__.count(".") >= 2
    assert "__version__" in dir(chat_rag)


def test_the_package_ships_its_typing_marker():
    """The annotations this file checks are only worth checking if a consumer's
    type checker can see them, and ``py.typed`` is what makes them visible."""
    assert (REPO / "src" / "chat_rag" / "py.typed").is_file()


# ------------------------------------------------------------- the signatures
def _public_callables():
    """Every callable a consumer can reach: the published functions, and the
    public methods and properties of the published classes."""
    for name in sorted(api.__all__):
        member = getattr(api, name)
        if inspect.isfunction(member):
            yield name, member, member
            continue
        if not inspect.isclass(member) or issubclass(member, BaseException):
            continue
        for attribute, value in sorted(vars(member).items()):
            if attribute.startswith("_") and attribute != "__init__":
                continue
            target = value.fget if isinstance(value, property) else value
            if inspect.isfunction(target):
                yield f"{name}.{attribute}", target, member


def _names_in(annotation) -> set[str]:
    """Every type named by an annotation, however deeply it is nested."""
    found: set[str] = set()

    def walk(node) -> None:
        if node is None or node is type(None):
            return
        origin = typing.get_origin(node)
        if origin is not None:
            walk(origin)
            for argument in typing.get_args(node):
                walk(argument)
            return
        if isinstance(node, type):
            found.add(f"{node.__module__}.{node.__qualname__}")

    walk(annotation)
    return found


def _published() -> set[str]:
    """Every published type, by the dotted name an annotation resolves to.

    By identity rather than by module, because where a published type *lives*
    is not the question. :class:`~chat_rag.config.Settings` is published and
    sits in ``chat_rag.config``; naming it in a signature is the surface doing
    its job, not a leak.
    """
    return {f"{member.__module__}.{member.__qualname__}"
            for member in (getattr(api, name) for name in api.__all__)
            if inspect.isclass(member)}


def _is_internal(dotted: str) -> bool:
    if dotted in _published() or _is_configuration(dotted):
        return False
    return any(dotted == package or dotted.startswith(package + ".")
               for package in INTERNAL_PACKAGES)


def _is_configuration(dotted: str) -> bool:
    return dotted.startswith(CONFIGURATION_PACKAGE + ".")


@pytest.mark.parametrize("member", [name for name, _, _ in _public_callables()])
def test_no_internal_type_appears_in_a_public_signature(member):
    """One parameter per published callable.

    Read from the resolved annotations rather than from the source, because a
    module using ``from __future__ import annotations`` writes them as strings
    and a text search would miss ``-> Services`` written as ``"Services"``.
    """
    target = next(f for name, f, _ in _public_callables() if name == member)
    try:
        hints = typing.get_type_hints(target)
    except Exception as unresolved:  # pragma: no cover - a broken annotation
        pytest.fail(f"{member} has an annotation that does not resolve: {unresolved}")

    leaks = {dotted for annotation in hints.values()
             for dotted in _names_in(annotation) if _is_internal(dotted)}
    if leaks and member in DECLARED_ESCAPES:
        return  # declared above, with the reason
    assert leaks == set(), (
        f"{member} names {sorted(leaks)} in its signature. Publish the type, "
        f"return a value from chat_rag.api.results, or declare the escape in "
        f"DECLARED_ESCAPES with the reason.")


def test_only_value_objects_reach_a_public_signature_from_the_configuration():
    """What keeps the configuration exemption from becoming a hole.

    ``chat_rag.config`` is exempt because it is value objects, so this is the
    check that it still is: every type from it that a consumer can meet has to
    be a dataclass. The moment one of them is a manager, a session or anything
    holding a connection, this fails and the exemption has to be re-argued
    rather than inherited.
    """
    import dataclasses

    for member, target, _owner in _public_callables():
        for annotation in typing.get_type_hints(target).values():
            for dotted in _names_in(annotation):
                if not _is_configuration(dotted):
                    continue
                module, _, qualname = dotted.rpartition(".")
                found = getattr(__import__(module, fromlist=[qualname]), qualname)
                assert dataclasses.is_dataclass(found), (
                    f"{member} names {dotted}, which is not a value object")


def test_every_declared_escape_is_still_one():
    """The other direction: an escape that stopped leaking should stop being
    declared, so the list keeps meaning something."""
    stale = []
    for member in DECLARED_ESCAPES:
        target = next((f for name, f, _ in _public_callables() if name == member), None)
        if target is None:
            stale.append(f"{member} (no such public member)")
            continue
        hints = typing.get_type_hints(target)
        leaks = {dotted for annotation in hints.values()
                 for dotted in _names_in(annotation) if _is_internal(dotted)}
        if not leaks:
            stale.append(f"{member} (no longer names an internal type)")
    assert stale == [], f"declared escapes that are no longer needed: {stale}"


def test_no_public_callable_demands_an_internal_type():
    """The direction that would leave a caller with no API at all.

    A *return* can be an escape hatch -- a caller may ignore it. A
    **parameter** cannot: a method that requires a ``Services`` is a method
    only this repository can call, and no declaration makes that acceptable.
    """
    offenders = []
    for member, target, _owner in _public_callables():
        hints = typing.get_type_hints(target)
        for parameter, annotation in hints.items():
            if parameter == "return":
                continue
            leaks = {d for d in _names_in(annotation) if _is_internal(d)}
            if leaks:
                offenders.append(f"{member}({parameter}: {sorted(leaks)})")
    assert offenders == [], f"public parameters demanding internal types: {offenders}"
