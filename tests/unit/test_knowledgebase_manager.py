from __future__ import annotations

import json

import pytest

from components.knowledgebase.manager import KnowledgeBaseManager


def test_api_create_contract_persists_legacy_and_v4_across_reload(tmp_path):
    store = tmp_path / "knowledge-bases.json"
    manager = KnowledgeBaseManager(str(store))

    legacy = manager.create_from_payload(
        {
            "name": "legacy-kb",
            "chunker": {"type": "legacy", "params": {}},
        }
    )
    v4 = manager.create_from_payload(
        {
            "name": "v4-kb",
            "chunker": {"type": "v4", "params": {}},
        }
    )

    persisted = json.loads(store.read_text(encoding="utf-8"))
    assert persisted[legacy["kb_id"]]["chunker"] == {
        "type": "legacy",
        "params": {},
    }
    assert persisted[v4["kb_id"]]["chunker"] == {
        "type": "v4",
        "params": {},
    }

    reloaded = KnowledgeBaseManager(str(store))
    assert reloaded.get(legacy["kb_id"])["chunker"]["type"] == "legacy"
    assert reloaded.get(v4["kb_id"])["chunker"] == {
        "type": "v4",
        "params": {},
    }


def test_missing_chunker_defaults_to_canonical_legacy(tmp_path):
    manager = KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json"))

    created = manager.create_from_payload({"name": "default-kb"})

    assert created["chunker"] == {"type": "legacy", "params": {}}


def test_v4_runtime_params_are_rejected_at_persistence_boundary(tmp_path):
    manager = KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json"))

    with pytest.raises(ValueError, match="accepts no runtime"):
        manager.create_from_payload(
            {
                "name": "invalid-v4",
                "chunker": {"type": "v4", "params": {"max_tokens": 256}},
            }
        )
