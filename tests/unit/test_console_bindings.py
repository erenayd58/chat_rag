"""Every control the console's JavaScript binds must exist in a template.

This is here because a translation pass once renamed element ids: "Save" and
"Cancel" were replaced wherever they appeared, including inside ``id="..."``,
so ``#settingsSave`` became ``#settingsKaydet`` while the handler kept binding
the old name. ``$('#settingsSave').addEventListener`` then threw at module
load and every line after it -- including the page's own initialisation --
never ran, which showed up as a knowledge-base page stuck on its spinner.

The lesson is not "add a null guard": a handler that finds nothing is a
control nobody can use, and hiding the error hides the broken control. The
binding and the markup have to agree, and this test is what says so.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "templates"
SCRIPTS = ROOT / "static" / "js"

#: Ids the scripts create themselves rather than find in the markup.
CREATED_IN_JS = {"typingIndicator"}


def _template_ids() -> set[str]:
    ids: set[str] = set()
    for path in TEMPLATES.glob("*.html"):
        ids |= set(re.findall(r'\bid="([^"]+)"', path.read_text(encoding="utf-8")))
    return ids


def _bound_ids(text: str) -> set[str]:
    """Ids this script looks up by name, however it spells the lookup."""
    found = set(re.findall(r"""\$\(['"]#([A-Za-z0-9_-]+)['"]\)""", text))
    found |= set(re.findall(r"""getElementById\(['"]([A-Za-z0-9_-]+)['"]""", text))
    found |= set(re.findall(r"""querySelector\(['"]#([A-Za-z0-9_-]+)['"]\)""", text))
    return found - CREATED_IN_JS


@pytest.mark.parametrize("script", sorted(SCRIPTS.glob("*.js")), ids=lambda p: p.name)
def test_every_bound_control_exists_in_the_markup(script):
    missing = sorted(_bound_ids(script.read_text(encoding="utf-8")) - _template_ids())
    assert not missing, (
        f"{script.name} binds controls that no template defines: {missing}. "
        "A handler that finds nothing is a control the user cannot use -- fix "
        "the id or remove the handler, do not guard the lookup."
    )


def test_the_ids_a_broken_rename_would_take_are_still_there():
    """The four that were actually lost, named so a regression is obvious."""
    ids = _template_ids()
    for control in ("settingsSave", "statChunks", "uploadCancel", "confirmCancel"):
        assert control in ids, f"{control} is gone from the templates again"


def test_the_upload_dialog_offers_methods_rather_than_a_single_mode():
    """The multi-method picker is markup the upload script depends on."""
    detail = (TEMPLATES / "kb_detail.html").read_text(encoding="utf-8")
    assert 'id="methodList"' in detail and 'id="methodHint"' in detail
    assert "chunkMode" not in detail, "the retired single-mode radio is back"
