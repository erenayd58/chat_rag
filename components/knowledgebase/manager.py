# /Users/murseltasgin/projects/chat_rag/components/knowledgebase/manager.py
"""
Knowledge Base manager with JSON persistence.
Each KB defines: name, chunker config, embedding model, vector db provider/path, retrieval method defaults.
"""
import os
import json
import shutil
import uuid
from typing import Dict, Any, List, Optional

from config import paths


from components.chunker import registry as chunker_registry


def normalize_chunker_config(chunker: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the canonical persisted chunker selection.

    The names and aliases are the indexing registry's
    (``components/chunker/registry.py``), the same table the factory builds
    from and the options endpoint offers.
    """
    if chunker is None:
        return {"type": chunker_registry.DEFAULT_ID, "params": {}}
    if not isinstance(chunker, dict):
        raise ValueError("chunker must be an object")

    resolved = chunker_registry.resolve(chunker.get("type"))
    if resolved is None:
        raise ValueError(f"chunker.type must be {chunker_registry.expected()}")

    params = chunker.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("chunker.params must be an object")
    if params and not resolved.accepts_params:
        raise ValueError(f"{resolved.params_refusal} chunker params")
    return {"type": resolved.id, "params": dict(params)}


def _ensure_parent(path: str) -> None:
    """Create the directory a store file lives in, if it is not there yet.

    Historically these files sat in the working directory, which always
    exists. A configured data directory does not, until something makes it.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


class KnowledgeBaseManager:
    def __init__(self, store_path: Optional[str] = None):
        # Defaults to the historical file unless a data directory is set.
        self.store_path = store_path or paths.knowledge_bases()
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
        _ensure_parent(self.store_path)
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
        # Validate everything *before* an id is minted or anything is
        # persisted. A rejected payload used to be able to leave a half record
        # behind, and repeated attempts produced several knowledge bases with
        # the same name pointing at one empty store.
        name = str(name or "").strip()
        if not name:
            raise ValueError("name is required")
        if self.find_by_name(name) is not None:
            raise ValueError(f"A knowledge base named {name!r} already exists")
        provider = str(vector_db_provider or "chroma").strip().lower()
        if provider not in {"chroma", "faiss"}:
            raise ValueError("vector_db_provider must be 'chroma' or 'faiss'")
        cfg = {
            "name": name,
            "chunker": normalize_chunker_config(chunker),
            "embedding_model_name": embedding_model_name,
            "vector_db_provider": provider,
            "vector_db_path": vector_db_path,
            "retrieval_method": retrieval_method,
            "extra": extra or {}
        }

        kb_id = str(uuid.uuid4())[:8]
        while kb_id in self.kbs:
            kb_id = str(uuid.uuid4())[:8]
        self.kbs[kb_id] = cfg
        try:
            self._save()
        except Exception:
            # Never leave a record that only exists in memory: the next reload
            # would drop it and the store directory would be orphaned.
            self.kbs.pop(kb_id, None)
            raise
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

    def find_by_name(self, name: str) -> Optional[str]:
        """kb_id of the knowledge base with this name, comparing case- and
        space-insensitively."""
        wanted = " ".join(str(name or "").split()).casefold()
        if not wanted:
            return None
        for kb_id, cfg in self.kbs.items():
            if " ".join(str(cfg.get("name") or "").split()).casefold() == wanted:
                return kb_id
        return None

    def storage_path(self, kb_id: str, root: str = ".") -> Optional[str]:
        """Where this knowledge base keeps its vectors.

        Mirrors the resolution the app applies when it builds the pipeline: an
        explicit ``vector_db_path`` wins, otherwise the store is named after
        the knowledge base under the provider's root.
        """
        cfg = self.kbs.get(kb_id)
        if cfg is None:
            return None
        configured = cfg.get("vector_db_path")
        if configured:
            return os.path.abspath(os.path.join(root, configured))
        default = paths.vector_store(cfg.get("vector_db_provider") or "chroma", kb_id)
        return os.path.abspath(os.path.join(root, default))

    #: Roots and the fallback store the app uses when no knowledge base is
    #: selected. Deleting one of these would take every other store with it.
    PROTECTED_DIRECTORY_NAMES = ("chroma_db", "faiss_db")

    def _protected_roots(self, root: str) -> List[str]:
        """The shared roots, wherever this deployment puts them."""
        resolved = [
            os.path.abspath(os.path.join(root, paths.vector_store_root(provider)))
            for provider in ("chroma", "faiss")
        ]
        resolved += [
            os.path.abspath(os.path.join(root, name))
            for name in self.PROTECTED_DIRECTORY_NAMES
        ]
        return resolved

    def _is_protected(self, path: str, root: str) -> bool:
        return os.path.abspath(path) in self._protected_roots(root)

    def delete(self, kb_id: str) -> bool:
        """Remove the config record only. Storage is left in place."""
        if kb_id in self.kbs:
            self.kbs.pop(kb_id)
            self._save()
            return True
        return False

    def delete_with_storage(self, kb_id: str, root: str = ".") -> Dict[str, Any]:
        """Remove the record and, when nothing else needs it, its vector store.

        Several records can point at one directory -- that is exactly how eight
        knowledge bases came to share a single empty store here -- so the
        directory goes only once the last record referencing it is gone.
        """
        if kb_id not in self.kbs:
            return {"deleted": False, "reason": "not found"}

        path = self.storage_path(kb_id, root)
        note = None
        removed = False

        # Clear the store *before* the record. If the directory cannot go --
        # on Windows an open ChromaDB handle is enough to stop it -- the record
        # stays too, so the two never drift apart. Dropping the record first
        # would leave exactly the orphaned store this method exists to prevent.
        if not path:
            note = "no storage path"
        elif [o for o in self.kbs if o != kb_id and self.storage_path(o, root) == path]:
            note = "kept: still used by " + ", ".join(
                sorted(o for o in self.kbs if o != kb_id
                       and self.storage_path(o, root) == path)
            )
        elif self._is_protected(path, root):
            note = "kept: shared/default store"
        elif not os.path.isdir(path):
            note = "already absent"
        else:
            try:
                shutil.rmtree(path)
                removed = True
            except OSError as exc:
                return {
                    "deleted": False,
                    "kb_id": kb_id,
                    "storage_path": path,
                    "storage_removed": False,
                    "reason": f"vector store is in use and could not be removed: {exc}",
                }

        self.kbs.pop(kb_id)
        self._save()
        return {
            "deleted": True,
            "kb_id": kb_id,
            "storage_path": path,
            "storage_removed": removed,
            **({"storage_note": note} if note else {}),
        }


