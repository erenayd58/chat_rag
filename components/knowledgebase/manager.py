"""Knowledge Base manager, on PostgreSQL.

Each KB defines: name, chunker config, embedding model, vector db
provider/path, retrieval method defaults.

Step 8 moved the records from ``.knowledge_bases.json`` into the
``knowledge_bases`` table. The class did not move: every caller -- the use
cases, both HTTP surfaces, the CLI and the Viewer's workspace snapshot -- asks
this object the same questions it always did, and gets the same dictionaries
back. What changed underneath is worth stating, because two of the rules this
class enforces were only ever true for a single process:

* **A name is unique.** It used to be checked by scanning the in-memory
  records and then writing the file, which two requests could both pass
  before either wrote. It is now a unique index on the normalised name, so
  the second creation fails on the constraint and is reported as the same
  ``ValueError`` a caller already handles.
* **A record either exists or does not.** The file store wrote the whole
  document on every change, so an interrupted write was the failure mode it
  spent most of its code defending against. A row is written in a
  transaction; there is no half-written store to recover from.

The vector store is still a directory (:meth:`storage_path`,
:meth:`delete_with_storage`); that is Step 9's problem, and deleting a
knowledge base is still two acts in a fixed order -- the store first, so a
directory that cannot be removed leaves the record in place rather than
orphaning itself.
"""
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

from config import paths
from storage import KnowledgeBaseRepository, session_scope

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


def _is_duplicate_name(error: Exception) -> bool:
    """Did this failure come from the unique index on the name?

    Asked of a real ``IntegrityError`` rather than guessed from a prior read,
    which is the whole difference between "unique because we looked" and
    "unique because it cannot be otherwise".
    """
    return "uq_knowledge_bases_name_key" in str(getattr(error, "orig", error))


class KnowledgeBaseManager:
    def __init__(self, store_path: Optional[str] = None):
        """``store_path`` is accepted and unused.

        It named the JSON file these records lived in until Step 8. Callers
        that still pass one -- the contract suites point every store at their
        own ``tmp_path`` -- keep working; the records come from PostgreSQL
        either way, and the value is kept only so a diagnostic can say what a
        caller thought it was opening.
        """
        self.store_path = store_path or paths.knowledge_bases()

    # ------------------------------------------------------------- reading
    @property
    def kbs(self) -> Dict[str, Dict[str, Any]]:
        """Every record, keyed by id. Read from the database on each access:
        another process may have created one since the last look, and the
        file-era habit of caching this in the instance is exactly how two
        requests came to write over each other."""
        with session_scope() as session:
            return KnowledgeBaseRepository(session).all()

    def list(self) -> List[Dict[str, Any]]:
        return [{"kb_id": kb_id, **cfg} for kb_id, cfg in self.kbs.items()]

    def get(self, kb_id: str) -> Optional[Dict[str, Any]]:
        with session_scope() as session:
            return KnowledgeBaseRepository(session).get(kb_id)

    def find_by_name(self, name: str) -> Optional[str]:
        """kb_id of the knowledge base with this name, comparing case- and
        space-insensitively."""
        with session_scope() as session:
            return KnowledgeBaseRepository(session).find_by_name(name)

    # ------------------------------------------------------------- writing
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
        provider = str(vector_db_provider or "chroma").strip().lower()
        if provider != "chroma":
            raise ValueError("vector_db_provider must be 'chroma'")
        cfg = {
            "name": name,
            "chunker": normalize_chunker_config(chunker),
            "embedding_model_name": embedding_model_name,
            "vector_db_provider": provider,
            "vector_db_path": vector_db_path,
            "retrieval_method": retrieval_method,
            "extra": extra or {}
        }

        from sqlalchemy.exc import IntegrityError

        # One transaction: the row is there or the name is still free. The
        # duplicate check is the unique index, not a read followed by a write
        # -- two requests creating "Yillik raporlar" at the same moment used to
        # both pass the read.
        try:
            with session_scope() as session:
                repository = KnowledgeBaseRepository(session)
                kb_id = str(uuid.uuid4())[:8]
                while repository.get(kb_id) is not None:
                    kb_id = str(uuid.uuid4())[:8]
                record = repository.create(kb_id, cfg)
        except IntegrityError as error:
            if _is_duplicate_name(error):
                raise ValueError(
                    f"A knowledge base named {name!r} already exists"
                ) from error
            raise
        return {"kb_id": kb_id, **record}

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
        updates = dict(updates)
        if "chunker" in updates:
            updates["chunker"] = normalize_chunker_config(updates["chunker"])
        with session_scope() as session:
            record = KnowledgeBaseRepository(session).update(kb_id, updates)
        if record is None:
            return None
        return {"kb_id": kb_id, **record}

    def storage_path(self, kb_id: str, root: str = ".") -> Optional[str]:
        """Where this knowledge base keeps its vectors.

        Mirrors the resolution the app applies when it builds the pipeline: an
        explicit ``vector_db_path`` wins, otherwise the store is named after
        the knowledge base under the provider's root.
        """
        cfg = self.get(kb_id)
        if cfg is None:
            return None
        return self._storage_path_of(kb_id, cfg, root)

    @staticmethod
    def _storage_path_of(kb_id: str, cfg: Dict[str, Any], root: str) -> str:
        configured = cfg.get("vector_db_path")
        if configured:
            return os.path.abspath(os.path.join(root, configured))
        return os.path.abspath(os.path.join(root, paths.vector_store(kb_id)))

    #: The root and the fallback store the app uses when no knowledge base
    #: is selected. Deleting one of these would take every store with it.
    PROTECTED_DIRECTORY_NAMES = ("chroma_db",)

    def _protected_roots(self, root: str) -> List[str]:
        """The shared roots, wherever this deployment puts them."""
        resolved = [os.path.abspath(os.path.join(root, paths.vector_store_root()))]
        resolved += [
            os.path.abspath(os.path.join(root, name))
            for name in self.PROTECTED_DIRECTORY_NAMES
        ]
        return resolved

    def _is_protected(self, path: str, root: str) -> bool:
        return os.path.abspath(path) in self._protected_roots(root)

    def delete(self, kb_id: str) -> bool:
        """Remove the config record only. Storage is left in place."""
        with session_scope() as session:
            return KnowledgeBaseRepository(session).delete(kb_id)

    def delete_with_storage(self, kb_id: str, root: str = ".") -> Dict[str, Any]:
        """Remove the record and, when nothing else needs it, its vector store.

        Several records can point at one directory -- that is exactly how eight
        knowledge bases came to share a single empty store here -- so the
        directory goes only once the last record referencing it is gone.

        The record is read and deleted inside one transaction, with the row
        locked for the length of it, so two deletions of the same knowledge
        base cannot both decide they are the last owner of its directory. The
        second waits, then finds no record and reports "not found" -- rather
        than both removing the same directory, which is what happened when the
        read and the write were two acts with a file rewrite between them.
        """
        with session_scope() as session:
            repository = KnowledgeBaseRepository(session)
            cfg = repository.get(kb_id, lock=True)
            if cfg is None:
                return {"deleted": False, "reason": "not found"}
            records = repository.all()

            path = self._storage_path_of(kb_id, cfg, root)
            sharers = sorted(
                other for other, other_cfg in records.items()
                if other != kb_id and self._storage_path_of(other, other_cfg, root) == path
            )
            note = None
            removed = False

            # Clear the store *before* the record. If the directory cannot go
            # -- on Windows an open ChromaDB handle is enough to stop it -- the
            # record stays too, so the two never drift apart. Dropping the
            # record first would leave exactly the orphaned store this method
            # exists to prevent, and here it would also commit that decision.
            if not path:
                note = "no storage path"
            elif sharers:
                note = "kept: still used by " + ", ".join(sharers)
            elif self._is_protected(path, root):
                note = "kept: shared/default store"
            elif not os.path.isdir(path):
                note = "already absent"
            else:
                try:
                    shutil.rmtree(path)
                    removed = True
                except OSError as exc:
                    # The transaction is abandoned by raising nothing and
                    # returning: nothing was written in it, so there is
                    # nothing to undo, and the record is still there.
                    return {
                        "deleted": False,
                        "kb_id": kb_id,
                        "storage_path": path,
                        "storage_removed": False,
                        "reason": f"vector store is in use and could not be removed: {exc}",
                    }

            repository.delete(kb_id)
            return {
                "deleted": True,
                "kb_id": kb_id,
                "storage_path": path,
                "storage_removed": removed,
                **({"storage_note": note} if note else {}),
            }
