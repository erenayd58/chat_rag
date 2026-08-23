# /Users/murseltasgin/projects/chat_rag/components/knowledgebase/manager.py
"""
Knowledge Base manager with JSON persistence.
Each KB defines: name, chunker config, embedding model, vector db provider/path, retrieval method defaults.
"""
import os
import json
import uuid
from typing import Dict, Any, List, Optional


_LEGACY_CHUNKER_NAMES = {"legacy", "semanticchunker", "semantic_chunker"}
_V4_CHUNKER_NAMES = {"v4", "frozenv4chunker", "frozen_v4_chunker"}
_STRUCTURAL_CHUNKER_NAMES = {
    "structure_first",
    "structurefirst",
    "structural",
    "structuralchunker",
    "structural_chunker",
}


def normalize_chunker_config(chunker: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the canonical persisted chunker selection."""
    if chunker is None:
        return {"type": "legacy", "params": {}}
    if not isinstance(chunker, dict):
        raise ValueError("chunker must be an object")

    raw_type = str(chunker.get("type") or "").strip().lower()
    if raw_type in _LEGACY_CHUNKER_NAMES:
        chunker_type = "legacy"
    elif raw_type in _V4_CHUNKER_NAMES:
        chunker_type = "v4"
    elif raw_type in _STRUCTURAL_CHUNKER_NAMES:
        chunker_type = "structure_first"
    else:
        raise ValueError(
            "chunker.type must be 'legacy', 'v4' or 'structure_first'"
        )

    params = chunker.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("chunker.params must be an object")
    if chunker_type == "v4" and params:
        raise ValueError("Frozen V4 accepts no runtime chunker params")
    if chunker_type == "structure_first" and params:
        raise ValueError("Structure-first accepts no runtime chunker params")
    return {"type": chunker_type, "params": dict(params)}


class KnowledgeBaseManager:
    def __init__(self, store_path: str = "./.knowledge_bases.json"):
        self.store_path = store_path
        self.kbs: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.kbs = {
                        kb_id: {
                            **cfg,
                            "chunker": normalize_chunker_config(cfg.get("chunker")),
                        }
                        for kb_id, cfg in data.items()
                        if isinstance(cfg, dict)
                    }
            except Exception:
                self.kbs = {}

    def _save(self) -> None:
        tmp = self.store_path + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.kbs, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.store_path)

    def list(self) -> List[Dict[str, Any]]:
        return [
            {"kb_id": kb_id, **cfg}
            for kb_id, cfg in self.kbs.items()
        ]

    def get(self, kb_id: str) -> Optional[Dict[str, Any]]:
        return self.kbs.get(kb_id)

    def create(
        self,
        name: str,
        *,
        chunker: Dict[str, Any] = None,
        embedding_model_name: str = None,
        vector_db_provider: str = "chroma",
        vector_db_path: str = None,
        retrieval_method: str = "hybrid",
        extra: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        kb_id = str(uuid.uuid4())[:8]
        cfg = {
            "name": name,
            "chunker": normalize_chunker_config(chunker),
            "embedding_model_name": embedding_model_name,
            "vector_db_provider": vector_db_provider,
            "vector_db_path": vector_db_path,
            "retrieval_method": retrieval_method,
            "extra": extra or {}
        }
        self.kbs[kb_id] = cfg
        self._save()
        return {"kb_id": kb_id, **cfg}

    def create_from_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a KB from the exact JSON contract accepted by POST /api/kb."""
        if not isinstance(data, dict):
            raise ValueError("Knowledge-base payload must be an object")
        return self.create(
            name=str(data.get("name") or "").strip() or "Knowledge Base",
            chunker=data.get("chunker"),
            embedding_model_name=data.get("embedding_model_name"),
            vector_db_provider=data.get("vector_db_provider") or "chroma",
            vector_db_path=data.get("vector_db_path"),
            retrieval_method=data.get("retrieval_method") or "hybrid",
            extra=data.get("extra"),
        )

    def update(self, kb_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if kb_id not in self.kbs:
            return None
        updates = dict(updates)
        if "chunker" in updates:
            updates["chunker"] = normalize_chunker_config(updates["chunker"])
        self.kbs[kb_id].update(updates)
        self._save()
        return {"kb_id": kb_id, **self.kbs[kb_id]}

    def delete(self, kb_id: str) -> bool:
        if kb_id in self.kbs:
            self.kbs.pop(kb_id)
            self._save()
            return True
        return False


