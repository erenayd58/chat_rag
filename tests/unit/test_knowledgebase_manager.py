from __future__ import annotations

import json

import pytest

from components.knowledgebase.manager import KnowledgeBaseManager


def test_api_create_contract_persists_every_chunker_across_reload(tmp_path):
    store = tmp_path / "knowledge-bases.json"
    manager = KnowledgeBaseManager(str(store))

    structure_first = manager.create_from_payload(
        {
            "name": "structure-first-kb",
            "chunker": {"type": "structure_first", "params": {}},
        }
    )
    v4 = manager.create_from_payload(
        {
            "name": "v4-kb",
            "chunker": {"type": "v4", "params": {}},
        }
    )

    persisted = {row["kb_id"]: row for row in KnowledgeBaseManager(str(store)).list()}
    assert persisted[structure_first["kb_id"]]["chunker"] == {
        "type": "structure_first",
        "params": {},
    }
    assert persisted[v4["kb_id"]]["chunker"] == {
        "type": "v4",
        "params": {},
    }

    reloaded = KnowledgeBaseManager(str(store))
    assert reloaded.get(structure_first["kb_id"])["chunker"]["type"] == "structure_first"
    assert reloaded.get(v4["kb_id"])["chunker"] == {
        "type": "v4",
        "params": {},
    }


def test_missing_chunker_defaults_to_the_product_chunker(tmp_path):
    manager = KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json"))

    created = manager.create_from_payload({"name": "default-kb"})

    assert created["chunker"] == {"type": "structure_first", "params": {}}


def test_v4_runtime_params_are_rejected_at_persistence_boundary(tmp_path):
    manager = KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json"))

    with pytest.raises(ValueError, match="accepts no runtime"):
        manager.create_from_payload(
            {
                "name": "invalid-v4",
                "chunker": {"type": "v4", "params": {"max_tokens": 256}},
            }
        )


def test_structure_first_chunker_is_selectable():
    from components.knowledgebase.manager import normalize_chunker_config

    for alias in ("structure_first", "structural", "StructuralChunker"):
        assert normalize_chunker_config({"type": alias}) == {
            "type": "structure_first",
            "params": {},
        }


def test_structure_first_rejects_runtime_params():
    import pytest

    from components.knowledgebase.manager import normalize_chunker_config

    with pytest.raises(ValueError):
        normalize_chunker_config({"type": "structure_first", "params": {"chunk_size": 300}})


def test_unknown_chunker_still_rejected():
    import pytest

    from components.knowledgebase.manager import normalize_chunker_config

    with pytest.raises(ValueError):
        normalize_chunker_config({"type": "nope"})
