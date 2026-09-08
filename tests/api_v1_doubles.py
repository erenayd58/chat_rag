"""The two providers an end-to-end `/api/v1` test replaces, and the corpus it runs on.

Everything else in those tests is the real product -- the chunker, the store,
the retriever, the ingest workers, the packager, PostgreSQL and pgvector. Only
the answer model and the embedding model are here, because they are the two
things that would otherwise leave the machine, and an assertion that depends
on a provider being reachable is an assertion about somebody's network.

Both are deterministic. The same text produces the same vector in every run
and every process, which is what makes it possible to assert a retrieval
ordering at all; and the answer model cites a label it actually found in the
prompt, so ``grounded`` is true for the reason the product says it is true
rather than because a fixture hard-coded ``[S1]``.
"""

from __future__ import annotations

import hashlib

import numpy as np

from components.llm import BaseLLM

#: How long a job or a packaging run may take before a test calls it stuck.
#: Generous: these run real chunkers on real worker threads.
PATIENCE_SECONDS = 60.0


class DeterministicEmbedding:
    """An embedding with no network and no model file.

    A bag of hashed words, so two texts sharing words land near each other and
    a dense search has something to rank. The width is fixed, so the store's
    manifest is the same across every pipeline a test builds and a restart
    reads back a compatible index rather than a foreign one.
    """

    DIMENSION = 16

    def encode(self, texts, convert_to_tensor: bool = False, **kwargs):
        if isinstance(texts, str):
            return self._vector(texts)
        return np.vstack([self._vector(text) for text in texts])

    def encode_documents(self, texts) -> np.ndarray:
        return np.asarray(self.encode(list(texts)), dtype=float)

    def encode_queries(self, texts) -> np.ndarray:
        return self.encode_documents(texts)

    def get_name(self) -> str:
        return "deterministic-test-embedding"

    def get_dimension(self) -> int:
        return self.DIMENSION

    @classmethod
    def _vector(cls, text: str) -> np.ndarray:
        vector = np.zeros(cls.DIMENSION, dtype=float)
        for word in (text or "").casefold().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[digest[0] % cls.DIMENSION] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else np.full(cls.DIMENSION, cls.DIMENSION ** -0.5)


class CitingLLM(BaseLLM):
    """An answer model that cites the first source it was actually given."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, temperature=0.3, max_tokens=500, **kwargs) -> str:
        self.calls += 1
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        label = "[S1]" if "[S1]" in prompt else ""
        return f"Kaynaklara gore takipteki alacaklar azaldi. {label}".strip()

    def get_name(self) -> str:
        return "CitingLLM"

    def get_model_name(self) -> str:
        return "citing-test-model"


#: Two documents with no word in common, so "which one came back" is a
#: question a lexical search can answer without a tie-break.
DOCUMENT = """# Yillik Rapor

## Takipteki Alacaklar

Takipteki alacaklar 2024 yilinda belirgin sekilde azaldi. Karsilik orani
yil boyunca yuzde seksenin uzerinde kaldi ve tahsilat performansi guclendi.

## Sermaye Yeterliligi

Sermaye yeterlilik rasyosu yasal sinirin uzerinde seyretti. Cekirdek sermaye
kalemleri yil icinde artis gosterdi.
"""

OTHER_DOCUMENT = """# Surdurulebilirlik Raporu

## Karbon Ayak Izi

Karbon ayak izi olcumleri kapsam bir ve kapsam iki icin yil boyunca duzenli
olarak raporlandi. Yenilenebilir enerji kullanimi arttirildi.
"""
