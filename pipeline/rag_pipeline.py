"""The pipeline: one object that ingests a document and answers a question.

It builds the embedder, the store, the retriever and the answer model from
settings, and runs the two flows the product has. Retrieval is deterministic
in every profile -- no query rewriting, no reranking, no model call before the
answer -- so what reaches the answer model is a function of the question and
the index alone.
"""
import contextlib
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import Settings
from config.settings import DEFAULT_RETRIEVAL_PROFILE, RETRIEVAL_PROFILES
from components.llm import (
    BaseLLM, AzureOpenAILLM, OllamaLLM, UnavailableLLM, OpenAICompatibleLLM, FallbackLLM,
)
from components.embedding import (
    BaseEmbedding, SentenceTransformerEmbedding, OpenAICompatibleEmbedding,
)
from components.context import assemble_context
from config import paths as data_paths
from components.vectordb import BaseVectorDB, PgVectorStore
from components.chunker import BaseChunker, create_chunker
from components.retriever import (
    HybridRRFRetriever,
    BM25OnlyRetriever,
    NullEmbedding,
    BenchmarkAlignedEmbedding,
    BenchmarkAlignedRetriever,
)
from components.parsers import ParserFactory
from core.models import DocumentChunk, RetrievalResult
from core.exceptions import (
    ConfigurationException, IndexIncompatibleException, IngestInterrupted, LLMException,
    QueryOverloaded, QueryTimeout, RAGException, RESOURCE_CONTROL_EXCEPTIONS,
)
from components.ingest.limits import checkpoint
from components.observability import telemetry as T
from components.query.limits import LimitedAnswerModel
from utils.logger import get_logger, RAGLogger

logger = get_logger("RAGPipeline")


@contextlib.contextmanager
def _no_stage():
    """The branch that computes no vectors is not an embedding stage."""
    yield


def _source_pages(metadata: Optional[Dict[str, Any]]) -> Optional[List[Any]]:
    """Page numbers a chunk spans, read from whichever metadata key the
    active chunker wrote. Purely presentational: used only to label the
    sources returned with an answer."""
    md = metadata or {}
    try:
        pages = json.loads(md.get('pages_json') or '[]')
        if pages:
            return pages
    except (TypeError, ValueError):
        pass
    for key in ('page', 'page_number', 'page_start'):
        value = md.get(key)
        if value not in (None, ''):
            return [value]
    return None


class RAGPipeline:
    """Ingestion and query for one knowledge base."""

    def __init__(
        self,
        llm_model: Optional[BaseLLM] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        vector_db: Optional[BaseVectorDB] = None,
        chunker: Optional[BaseChunker] = None,
        settings: Optional[Settings] = None
    ):
        self.settings = settings or Settings()
        self.retrieval_profile = getattr(
            self.settings, 'retrieval_profile', DEFAULT_RETRIEVAL_PROFILE
        )
        if self.retrieval_profile not in RETRIEVAL_PROFILES:
            raise ValueError(
                "retrieval_profile must be one of "
                + ", ".join(sorted(RETRIEVAL_PROFILES))
            )

        self.llm_model = llm_model or self._create_llm()
        self.embedding_model = embedding_model or self._create_embedding()
        self.vector_db = vector_db or self._create_vectordb()
        self.chunker = chunker or self._create_chunker()

        if self.retrieval_profile == 'bm25_only':
            self.hybrid_retriever = BM25OnlyRetriever(
                self.embedding_model,
                self.vector_db,
            )
        elif self.retrieval_profile == 'hybrid_rrf':
            # The final product profile: stored dense vectors + frozen BM25,
            # fused by RRF. The store carries the embedding manifest.
            self.hybrid_retriever = HybridRRFRetriever(
                self.embedding_model,
                self.vector_db,
            )
        else:
            if not isinstance(self.embedding_model, BenchmarkAlignedEmbedding):
                raise ValueError(
                    "benchmark_aligned requires BenchmarkAlignedEmbedding"
                )
            self.hybrid_retriever = BenchmarkAlignedRetriever(
                self.embedding_model,
                self.vector_db,
            )

        self.parser_factory = ParserFactory()

        # Build the lexical index at startup from whatever the store holds.
        try:
            existing_chunks = self.vector_db.get_all_chunks()
            if existing_chunks:
                self.hybrid_retriever.build_keyword_index(existing_chunks)
        except Exception:
            # Non-fatal; keyword search will lazily build if needed
            pass

    @property
    def answer_model(self) -> BaseLLM:
        """The answer model as a query calls it: one budget slot per call,
        under the query deadline (see components/query/limits.py).

        A view over ``llm_model`` built on access rather than stored, so a
        test or a tool that swaps ``llm_model`` underneath is honoured and
        nothing is wrapped twice. The wrapper holds no state, so this costs
        an object per query and nothing else.
        """
        inner = self.llm_model
        if isinstance(inner, LimitedAnswerModel):
            return inner
        limits = getattr(getattr(self, 'settings', None), 'query_limits', None)
        wait = getattr(limits, 'answer_wait_seconds', None)
        return LimitedAnswerModel(inner, wait_seconds=wait)

    def _create_llm(self) -> BaseLLM:
        """The answer model: a primary provider and, when configured, a local
        fallback.

        A provider that cannot be reached does not stop the pipeline from
        being built. Ingestion, chunking, structural QA and lexical retrieval
        need no language model at all, so the failure is carried by
        UnavailableLLM and raised, with its cause, only when something asks
        for generated text.
        """
        primary = self._build_answer_model(
            provider=self.settings.answer_provider,
            model=self.settings.answer_model,
        )
        fallback_provider = getattr(self.settings, 'answer_fallback_provider', 'none') or 'none'
        if fallback_provider in ('', 'none', 'off', 'false'):
            return primary
        fallback = self._build_answer_model(
            provider=fallback_provider,
            model=getattr(self.settings, 'answer_fallback_model', ''),
        )
        return FallbackLLM(primary, fallback)

    def _build_answer_model(self, *, provider: str, model: str) -> BaseLLM:
        provider = (provider or '').strip().lower()
        try:
            if provider in ('openrouter', 'openai_compatible', 'openai-compatible'):
                endpoint = getattr(self.settings, 'answer_endpoint', '') or None
                kwargs = {
                    'api_key_env': getattr(self.settings, 'answer_api_key_env', 'OPENROUTER_API_KEY'),
                    'timeout_seconds': float(getattr(self.settings, 'answer_timeout', 120.0)),
                }
                if endpoint:
                    kwargs['endpoint'] = endpoint
                return OpenAICompatibleLLM(model, **kwargs)
            if provider == 'ollama':
                return OllamaLLM(
                    model=model or self.settings.ollama_model,
                    base_url=self.settings.ollama_base_url,
                    timeout=self.settings.ollama_timeout,
                )
            if provider == 'azure':
                return AzureOpenAILLM(
                    endpoint=self.settings.azure_endpoint,
                    api_key=self.settings.azure_api_key,
                    deployment=model or self.settings.azure_deployment,
                    api_version=self.settings.azure_api_version,
                )
            raise ConfigurationException(f"unknown answer provider {provider!r}")
        except Exception as exc:
            endpoint = (
                self.settings.ollama_base_url if provider == 'ollama'
                else self.settings.azure_endpoint if provider == 'azure'
                else getattr(self.settings, 'answer_endpoint', '') or 'the configured gateway'
            )
            return UnavailableLLM(provider or 'answer model', str(exc), endpoint)

    def _create_embedding(self) -> BaseEmbedding:
        if self.retrieval_profile == 'bm25_only':
            # No dense leg in this profile: never load an embedding model.
            return NullEmbedding()
        if self.retrieval_profile == 'benchmark_aligned':
            return BenchmarkAlignedEmbedding()
        provider = (getattr(self.settings, 'embedding_provider', '') or 'sentence_transformers').lower()
        if provider in ('openrouter', 'openai_compatible', 'openai-compatible'):
            kwargs = {
                'api_key_env': getattr(self.settings, 'embedding_api_key_env', 'OPENROUTER_API_KEY'),
                'batch_size': int(getattr(self.settings, 'embedding_batch_size', 32)),
                'timeout_seconds': float(getattr(self.settings, 'embedding_timeout', 120.0)),
                'dimensions': getattr(self.settings, 'embedding_dimensions', None),
                'concurrency': int(getattr(self.settings, 'embedding_concurrency', 4)),
                'cache_dir': data_paths.embedding_cache(),
            }
            endpoint = getattr(self.settings, 'embedding_endpoint', '') or None
            if endpoint:
                kwargs['endpoint'] = endpoint
            return OpenAICompatibleEmbedding(self.settings.embedding_model_name, **kwargs)
        return SentenceTransformerEmbedding(
            model_name=self.settings.embedding_model_name
        )

    def _create_vectordb(self) -> BaseVectorDB:
        """The collection this pipeline searches, in PostgreSQL.

        Named, not opened: constructing a store runs no statement, so building
        a pipeline still works on a machine with no reachable database and the
        failure arrives at the first call that actually needs a row.
        """
        return PgVectorStore(
            collection=self.settings.vector_collection,
            kb_id=getattr(self.settings, 'vector_kb_id', None),
        )

    def _create_chunker(self) -> BaseChunker:
        return create_chunker(self.settings)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document_from_file(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        doc_title: Optional[str] = None,
        additional_metadata: Dict[str, Any] = None,
        deep_analysis: bool = False
    ) -> List[DocumentChunk]:
        """Parse a file and ingest it.

        ``deep_analysis`` is the per-upload ingest mode: False is the Standard
        structure-only path, True runs the Deep Analysis pipeline. Never a
        query-time decision.
        """
        try:
            import os

            if doc_id is None:
                doc_id = os.path.basename(file_path).replace('.', '_')

            if doc_title is None:
                doc_title = os.path.basename(file_path)

            print(f"Parsing file: {file_path}")

            from time import perf_counter
            _parse_started = perf_counter()

            # One stage covering everything the parser does for this file: the
            # text, the structured units and the metadata. Measured once, so
            # the number here and the one the Viewer shows are the same number.
            with T.stage(T.PARSE):
                document_text = self.parser_factory.parse_file(file_path)

                # Structure-aware chunkers need typed units (heading/list/table)
                # with section_path and page provenance; without them every
                # structural setting silently degrades to token-budget cutting.
                try:
                    parsed_units = self.parser_factory.parse_units(file_path)
                except Exception as exc:
                    print(f"  - Structured unit extraction unavailable: {exc}")
                    parsed_units = None

                parser_metadata = self.parser_factory.get_metadata(file_path)
                self.last_parse_seconds = round(perf_counter() - _parse_started, 2)
                T.annotate(characters=len(document_text or ''),
                           units=len(parsed_units) if parsed_units else 0)

            # A stage boundary: the file is parsed and nothing is written.
            # If this ingest is a job that has run out of time or been
            # cancelled, this is where it stops.
            checkpoint()

            merged_metadata = additional_metadata or {}
            merged_metadata.update(parser_metadata)

            print(f"  - Extracted {len(document_text)} characters")
            print(f"  - Parser used: {parser_metadata.get('parser', 'unknown')}")
            if parsed_units is not None:
                print(f"  - Structured canonical units: {len(parsed_units)}")
            else:
                print("  - Structured canonical units: none (flat text fallback)")

            return self.ingest_document(
                document_text=document_text,
                doc_id=doc_id,
                doc_title=doc_title,
                additional_metadata=merged_metadata,
                parsed_units=parsed_units,
                deep_analysis=deep_analysis
            )

        except (ConfigurationException, IndexIncompatibleException, IngestInterrupted):
            # A misconfigured or unsupported Deep Analysis request, a store
            # that must be re-indexed first, or a job stopped on purpose must
            # reach the caller as what it is, not wrapped into a generic failure.
            raise
        except Exception as e:
            raise RAGException(f"Failed to ingest document from file {file_path}: {e}")

    def ingest_document(
        self,
        document_text: str,
        doc_id: str,
        doc_title: str,
        additional_metadata: Dict[str, Any] = None,
        parsed_units: Optional[List[Dict[str, Any]]] = None,
        deep_analysis: bool = False
    ) -> List[DocumentChunk]:
        """Chunk, embed and index one already-parsed document.

        ``deep_analysis`` runs ``amsc.deep.pipeline`` instead of the plain
        structural walk. It requires the structure-first chunker; a missing or
        failing model provider degrades to the deterministic quality contract
        and is reported in the status, never raised. Everything after chunking
        is the unchanged shared path.
        """
        # The Deep Analysis report of the most recent ingest, for the caller
        # to persist into document metadata. Reset per ingest; stays None on
        # the Standard path.
        self.last_deep_analysis_report = None
        try:
            print(f"Ingesting document: {doc_title}")

            if deep_analysis:
                print("  - Creating chunks (Deep Analysis: amsc.deep.pipeline)...")
                if not hasattr(self.chunker, "chunk_text_deep"):
                    raise ConfigurationException(
                        "Deep Analysis requires the structure-first chunker; "
                        f"this knowledge base uses {self.chunker.get_name()}"
                    )
                from components.chunker.deep_analysis import (
                    build_configuration,
                    deep_config,
                    limited_providers,
                )
                from components.chunker.structural_chunker import (
                    HARD_MAX_TOKENS, MIN_TOKENS, SOFT_MAX_TOKENS, TARGET_TOKENS,
                )

                configuration = build_configuration(
                    self.settings,
                    deep_config(
                        min_tokens=MIN_TOKENS,
                        target_tokens=TARGET_TOKENS,
                        soft_max_tokens=SOFT_MAX_TOKENS,
                        hard_max_tokens=HARD_MAX_TOKENS,
                    ),
                )
                if configuration.missing:
                    print(f"  - {configuration.fallback_reason}")
                # The transports are built here, not inside amsc, so that
                # every proposer and verifier call goes through the
                # process-wide provider budget (components.ingest.limits).
                # (None, None) means no model run: the contract runs alone.
                provider, verifier_provider = limited_providers(configuration)
                with T.stage(T.DEEP):
                    chunks, deep_report = self.chunker.chunk_text_deep(
                        document_text, doc_id, doc_title, "",
                        configuration=configuration,
                        provider=provider,
                        verifier_provider=verifier_provider,
                        parser_metadata=additional_metadata,
                        parsed_units=parsed_units
                    )
                    T.annotate(chunks=len(chunks) if chunks else 0)
                self.last_deep_analysis_report = deep_report
                print(f"  - Deep Analysis status: {deep_report.get('status')}")
            else:
                print("  - Creating chunks...")
                with T.stage(T.CHUNK):
                    chunks = self.chunker.chunk_text(
                        document_text, doc_id, doc_title, "",
                        embedding_model=self.embedding_model,
                        parser_metadata=additional_metadata,
                        parsed_units=parsed_units
                    )
                    T.annotate(chunks=len(chunks) if chunks else 0)

            if not chunks:
                print("  ⚠️  Warning: No chunks created for document (text may be too short)")
                return []

            print(f"  - Created {len(chunks)} chunks")
            # Chunked, nothing written: a job that must stop, stops here.
            checkpoint()

            if self.retrieval_profile == 'hybrid_rrf':
                # Never add vectors of one embedding space to a store that
                # holds another: refuse, and let the console re-index.
                ok, reason = self.hybrid_retriever.can_accept_dense_writes()
                if not ok:
                    raise IndexIncompatibleException(
                        "This knowledge base's vector store must be re-indexed with the "
                        f"current embedding model before new documents can be added: {reason}"
                    )

            embeddings = []
            needs_embeddings = getattr(
                self.hybrid_retriever, "requires_document_embeddings", True
            )

            with T.stage(T.EMBED) if needs_embeddings else _no_stage():
                if not needs_embeddings:
                    # Lexical-only retrieval: no dense leg exists, so no vectors are
                    # computed and none are stored. Metadata still has to be applied
                    # because the chunk rows carry it into the store.
                    print("  - Skipping embeddings (lexical-only retrieval)")
                    if additional_metadata:
                        for chunk in chunks:
                            if chunk.metadata:
                                chunk.metadata.update(additional_metadata)
                else:
                    print("  - Generating embeddings...")
                    # A table embeds as its pipes *and* its rendering: the
                    # rendering is the half a question resembles, the markdown
                    # keeps the sentences and row labels it was derived from, and
                    # a chunk must never be searched for under less than its own
                    # text. Answer context still reads ``content`` alone.
                    matrix = self.embedding_model.encode_documents(
                        [chunk.retrieval_text for chunk in chunks]
                    )
                    for chunk, embedding in zip(chunks, matrix, strict=True):
                        chunk.embedding = embedding
                        embeddings.append(embedding.tolist())
                        if additional_metadata and chunk.metadata:
                            chunk.metadata.update(additional_metadata)
                T.annotate(vectors=len(embeddings))

            # The last seam before anything is written: past this point the
            # ingest runs to completion, because a half-written store is worse
            # than a late one.
            checkpoint()
            with T.stage(T.INDEX):
                print("  - Storing in vector database...")
                if chunks and (embeddings or not needs_embeddings):
                    self.vector_db.add_chunks(chunks, embeddings)
                    if self.retrieval_profile == 'hybrid_rrf' and embeddings:
                        self.hybrid_retriever.record_index(len(embeddings[0]))
                else:
                    print("  ⚠️  Warning: No chunks to store")
                    return []

                print("  - Building BM25 index...")
                all_chunks = self.vector_db.get_all_chunks()
                if all_chunks:
                    self.hybrid_retriever.build_keyword_index(all_chunks)
                T.annotate(stored=len(chunks), indexed=len(all_chunks or []))

            print(f"✓ Document '{doc_title}' ingested successfully!")
            return chunks

        except (ConfigurationException, IndexIncompatibleException, IngestInterrupted):
            # Deep Analysis misconfiguration, a stale vector store and a job
            # stopped on purpose are the caller's to handle explicitly;
            # wrapping them would read as a document failure.
            raise
        except Exception as e:
            raise RAGException(f"Document ingestion failed: {e}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = None
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """Run the question through the configured profile, as it was asked.

        Deterministic in every profile: no query rewriting, no reranking, no
        model call. The metadata says which legs actually answered, which is
        what lets a dense-unavailable knowledge base say so rather than
        silently returning keyword hits.
        """
        try:
            started = time.perf_counter()
            final_top_k = top_k if top_k is not None else self.settings.default_top_k
            results = self.hybrid_retriever.hybrid_search(query, top_k=final_top_k)
            rrf = self.hybrid_retriever.config.get("rrf")
            if self.retrieval_profile == 'hybrid_rrf':
                stats = dict(getattr(self.hybrid_retriever, 'last_stats', {}) or {})
                method = "hybrid_rrf" if stats.get("dense_used") else "bm25_only"
            else:
                stats = {}
                method = "bm25_only" if rrf is None else "equal_weight_rrf"
            metadata = {
                "original_query": query,
                "refined_query": query,
                "timestamp": datetime.now().isoformat(),
                "retrieval_profile": self.retrieval_profile,
                "retrieval_method": method,
                "query_expansion": False,
                "reranking": False,
                "contextualization": False,
                "candidate_pool_size": rrf["candidate_pool_size"] if rrf else None,
                "rrf_rank_constant": rrf["rank_constant"] if rrf else None,
            }
            if self.retrieval_profile == 'hybrid_rrf':
                stats.setdefault("latency_ms", round((time.perf_counter() - started) * 1000.0, 1))
                T.annotate(dense_hits=stats.get("dense_hits"), bm25_hits=stats.get("bm25_hits"),
                           fused=stats.get("fused_candidates"), dense_used=bool(stats.get("dense_used")))
                metadata["retrieval"] = stats
                metadata["dense_used"] = bool(stats.get("dense_used"))
                metadata["reindex_required"] = not bool(stats.get("dense_available"))
                metadata["dense_unavailable_reason"] = stats.get("dense_unavailable_reason")
            return results, metadata
        except RESOURCE_CONTROL_EXCEPTIONS:
            raise
        except Exception as e:
            raise RAGException(f"Retrieval failed: {e}") from e

    # ------------------------------------------------------------------
    # The final chain: grounded answers over assembled context
    # ------------------------------------------------------------------
    GROUNDED_SYSTEM_PROMPT = (
        "Sen bir kurumsal doküman asistanısın. Kullanıcının sorusunu YALNIZCA "
        "sana verilen kaynak parçalara dayanarak cevaplarsın.\n"
        "Kurallar:\n"
        "1. Yalnız [S1], [S2] ... etiketli kaynak parçalardaki bilgiyi kullan; dışarıdan "
        "bilgi ekleme, tahmin yürütme, kaynakta olmayan sayı veya tarih uydurma.\n"
        "2. Her iddianın sonuna dayandığı kaynağın etiketini köşeli parantezle yaz, "
        "örneğin [S1] veya [S2][S4]. Etiketleri değiştirme.\n"
        "3. Birden fazla kaynak gerekiyorsa bilgileri sentezle ve hepsini etiketle.\n"
        "4. Kaynaklarda açıkça yazan sayılardan hesap yapabilirsin: fark, toplam, oran "
        "veya yüzde puan farkı çıkarmak kaynağa dayalı bir cevaptır, uydurma değildir. "
        "Hesabın her girdisi kaynaklarda açıkça yazıyor olmalı; kullandığın sayıların "
        "etiketlerini yaz ve hesabı tek cümleyle göster. Girdilerden biri kaynakta "
        "yoksa hesap yapma.\n"
        "5. Sorunun bir bölümü kaynaklarda varsa o bölümü cevapla ve eksik kalanı açıkça "
        "söyle. Bir kısmını bilmiyorsun diye cevabın tamamını vermemek yanlıştır. Aynı "
        "şekilde kaynak bir olguyu veriyor ama gerekçesini vermiyorsa, olguyu aktar ve "
        "gerekçenin kaynaklarda belirtilmediğini söyle; gerekçe uydurma.\n"
        "6. Kaynaklar sorunun HİÇBİR bölümünü cevaplamaya yetmiyorsa cevabına tam olarak "
        "şu cümleyle başla: 'Kaynaklarda bu soru için yeterli bilgi yok.' ve ardından "
        "kaynaklarda bulunan en yakın bilgiyi kısaca belirt. Bu cümleyi yalnızca hiçbir "
        "bölüm cevaplanamıyorsa kullan; kısmen cevaplanabiliyorsa 5. kurala uy.\n"
        "7. Sorunun dilinde, doğal ve öz bir dille cevap ver; Türkçe soruya Türkçe cevap ver.\n"
        "8. Cevabı düz metin olarak yaz; JSON, başlık veya kaynak listesi ekleme.\n"
        "9. Çok dönemli bir tablodan (birden fazla yıl veya sütun içeren) sayı kullanırken "
        "sayının hangi döneme/sütuna ait olduğunu kaynakta doğrula ve cevabında belirt. Bir "
        "sayının ait olduğu dönem kesin olarak belirlenemiyorsa o sayıyı hiçbir döneme "
        "atfetme ve bunu söyle; yakın duruyor diye eşleştirme yapma. Bir kaynak parçada "
        "tablodan türetilmiş bir okuma bloğu varsa sayı-dönem eşleşmesinde onu esas al, "
        "alıntıyı yine kaynağın kendi metnine dayandır."
    )
    INSUFFICIENT_MARKERS = (
        "yeterli bilgi yok",
        "not enough information",
        "insufficient information",
    )
    _CITATION = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")

    def _cited_labels(self, answer: str) -> List[str]:
        found: List[str] = []
        for group in self._CITATION.findall(answer or ""):
            for label in re.findall(r"S\d+", group):
                if label not in found:
                    found.append(label)
        return found

    def _answer_grounded(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        metadata: Dict[str, Any],
        temperature: float,
        max_tokens: int,
        started: float,
    ) -> Dict[str, Any]:
        """Assemble a bounded, labelled context and ask the answer model for
        a cited answer. Provider failures reach the caller as LLMException
        (after the fallback, if one is configured); nothing here calls an
        ingest-time model."""
        retriever = self.hybrid_retriever
        with T.stage(T.CONTEXT, candidates=len(retrieval_results)):
            bundle = assemble_context(
                retrieval_results,
                max_tokens=int(getattr(self.settings, 'context_max_tokens', 3200)),
                max_sources=int(getattr(self.settings, 'context_max_sources', 8)),
                neighbor=getattr(retriever, 'neighbor', None),
                expand_neighbors=bool(getattr(self.settings, 'context_expand_neighbors', True)),
            )
            T.annotate(selected=len(bundle.sources), tokens=bundle.token_count)
        retrieval_ms = round((time.perf_counter() - started) * 1000.0, 1)

        user_prompt = (
            "Kaynak parçalar:\n\n"
            + (bundle.text or "(kaynak bulunamadı)")
            + "\n\nSoru: "
            + question
        )
        messages = [
            {"role": "system", "content": self.GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        answer_started = time.perf_counter()
        if not bundle.sources:
            answer = "Kaynaklarda bu soru için yeterli bilgi yok. Bu bilgi tabanında soruyla eşleşen bir parça bulunamadı."
            call: Dict[str, Any] = {"provider": None, "model": None, "fallback_used": False, "skipped": "no_sources"}
        else:
            # A cited, multi-source answer needs room, and a reasoning model
            # spends part of the budget thinking: the route's default of 500
            # truncated list-style answers mid-item in the live smoke.
            with T.stage(T.ANSWER):
                answer = self.answer_model.generate(
                    messages=messages, temperature=temperature, max_tokens=max(int(max_tokens), 1200)
                ).strip()
            call = dict(getattr(self.llm_model, 'last_call', None) or {
                "provider": getattr(self.llm_model, 'provider_id', self.llm_model.get_name()),
                "model": self.llm_model.get_model_name(),
                "fallback_used": False,
            })
        answer_ms = round((time.perf_counter() - answer_started) * 1000.0, 1)

        cited = [label for label in self._cited_labels(answer) if label in bundle.labels]
        lowered = answer.lower()
        sufficient = not any(marker in lowered for marker in self.INSUFFICIENT_MARKERS)

        sources = []
        for source in bundle.sources:
            row = source.as_dict()
            content = source.chunk.content or ""
            row["used"] = source.label in cited
            row["content_preview"] = content[:200] + "..." if len(content) > 200 else content
            row["content"] = content
            sources.append(row)

        identity = {}
        describe = getattr(self.embedding_model, 'describe', None)
        if callable(describe):
            identity = {k: v for k, v in describe().items() if k in ("provider", "model", "dimension", "fingerprint")}
        call.update({
            "latency_ms": answer_ms,
            "cited_labels": cited,
            "grounded": bool(cited),
            "sufficient": sufficient,
        })
        metadata.update({
            "context": {
                "selected": len(bundle.sources),
                "token_estimate": bundle.token_count,
                "max_tokens": bundle.max_tokens,
                "deduplicated": bundle.deduplicated,
                "expanded_neighbors": bundle.expanded,
                "dropped_over_budget": bundle.dropped_over_budget,
                "dropped_ranked_hits": bundle.dropped_ranked_hits,
            },
            "embedding": identity,
            "chunking_modes": sorted({row["chunking_mode"] for row in sources}),
            "answer": call,
            "latency_ms": {
                "retrieval": retrieval_ms,
                "answer": answer_ms,
                "total": round((time.perf_counter() - started) * 1000.0, 1),
            },
        })
        logger.info(
            "query: profile=%s method=%s dense_hits=%s bm25_hits=%s fused=%s context=%s tokens=%s "
            "answer=%s/%s fallback=%s cited=%s latency_ms=%s",
            self.retrieval_profile, metadata.get("retrieval_method"),
            (metadata.get("retrieval") or {}).get("dense_hits"),
            (metadata.get("retrieval") or {}).get("bm25_hits"),
            (metadata.get("retrieval") or {}).get("fused_candidates"),
            len(bundle.sources), bundle.token_count,
            call.get("provider"), call.get("model"), call.get("fallback_used"),
            ",".join(cited) or "-", metadata["latency_ms"]["total"],
        )
        return {"answer": answer, "sources": sources, "metadata": metadata}

    # ------------------------------------------------------------------
    # Index state
    # ------------------------------------------------------------------

    def embedding_index_status(self) -> Dict[str, Any]:
        """Whether the store's vectors belong to the current embedding."""
        retriever = self.hybrid_retriever
        if hasattr(retriever, 'refresh_index_status'):
            return retriever.refresh_index_status()
        dense = bool(getattr(retriever, 'requires_document_embeddings', True))
        return {
            "state": "not_applicable",
            "compatible": True,
            "dense_available": dense,
            "reason": f"The {self.retrieval_profile} profile does not track an embedding manifest.",
            "stored": {}, "current": {},
        }

    def reindex_embeddings(self) -> Dict[str, Any]:
        """Re-embed every stored chunk with the current model and rewrite
        the store's vectors. Texts and metadata are untouched; only the
        vectors -- and the manifest -- change."""
        if self.retrieval_profile != 'hybrid_rrf':
            raise ConfigurationException(
                f"Re-indexing applies to the hybrid_rrf profile, not {self.retrieval_profile}"
            )
        if not hasattr(self.vector_db, 'replace_all'):
            raise RAGException(f"{self.vector_db.get_name()} does not support re-indexing in place")
        started = time.perf_counter()
        chunks = self.vector_db.get_all_chunks()
        if not chunks:
            manifest = self.hybrid_retriever.record_index(int(self.embedding_model.get_dimension()))
            return {"chunks": 0, "dimension": manifest.get("embedding_dimension"),
                    "model": manifest.get("embedding_model"), "fingerprint": manifest.get("embedding_fingerprint"),
                    "seconds": round(time.perf_counter() - started, 2)}
        matrix = self.embedding_model.encode_documents(
            [chunk.retrieval_text for chunk in chunks]
        )
        vectors = [row.tolist() for row in matrix]
        self.vector_db.replace_all(chunks, vectors)
        self.hybrid_retriever.build_index(self.vector_db.get_all_chunks())
        manifest = self.hybrid_retriever.record_index(int(matrix.shape[1]))
        usage = getattr(self.embedding_model, 'last_usage', None)
        return {
            "chunks": len(chunks),
            "dimension": int(matrix.shape[1]),
            "model": manifest.get("embedding_model"),
            "provider": manifest.get("embedding_provider"),
            "fingerprint": manifest.get("embedding_fingerprint"),
            "seconds": round(time.perf_counter() - started, 2),
            "embedding_usage": (usage.__dict__ if hasattr(usage, '__dict__') else usage),
        }

    def model_chain(self) -> Dict[str, Any]:
        """The model roles as configured, plus the retrieval and context
        settings. Names and ids only; never a key."""
        from components.chunker.deep_analysis import build_configuration, deep_config
        from components.chunker.structural_chunker import (
            HARD_MAX_TOKENS, MIN_TOKENS, SOFT_MAX_TOKENS, TARGET_TOKENS,
        )
        deep = build_configuration(self.settings, deep_config(
            min_tokens=MIN_TOKENS, target_tokens=TARGET_TOKENS,
            soft_max_tokens=SOFT_MAX_TOKENS, hard_max_tokens=HARD_MAX_TOKENS,
        ))
        embedding: Dict[str, Any] = {"provider": None, "model": self.embedding_model.get_name()}
        describe = getattr(self.embedding_model, 'describe', None)
        if callable(describe):
            embedding = {k: v for k, v in describe().items() if k != 'endpoint'} | {"endpoint": describe().get("endpoint")}
        answer: Dict[str, Any]
        if hasattr(self.llm_model, 'describe'):
            answer = self.llm_model.describe()
        else:
            answer = {"primary": {"provider": getattr(self.llm_model, 'provider_id', self.llm_model.get_name()),
                                  "model": self.llm_model.get_model_name()}, "fallback": None}
        return {
            "agentic_chunking": {
                "model": deep.settings.proposer_model if deep.llm_available else None,
                "verifier_model": deep.settings.effective_verifier_model if deep.llm_available and deep.settings.verify else None,
                "endpoint": deep.settings.endpoint if deep.llm_available else None,
                "verify": deep.settings.verify,
                "llm_available": deep.llm_available,
                "missing": list(deep.missing),
                "entry_point": "amsc.deep.pipeline.chunk_document",
            },
            "embedding": embedding,
            "answer": answer,
            "retrieval": {
                "profile": self.retrieval_profile,
                "config": getattr(self.hybrid_retriever, 'config', None),
            },
            "context": {
                "max_tokens": int(getattr(self.settings, 'context_max_tokens', 3200)),
                "max_sources": int(getattr(self.settings, 'context_max_sources', 8)),
                "expand_neighbors": bool(getattr(self.settings, 'context_expand_neighbors', True)),
            },
        }

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------

    def get_retrieval_context(
        self,
        results: List[RetrievalResult],
        include_metadata: bool = True
    ) -> str:
        """Retrieved chunks as one plain-text context block."""
        context_parts = []

        for i, result in enumerate(results, 1):
            chunk = result.chunk
            context = f"[Result {i}] (Score: {result.score:.3f})\n"

            if include_metadata:
                context += f"Document: {chunk.doc_title}\n"
                if chunk.section_title:
                    context += f"Section: {chunk.section_title}\n"

            context += f"Content: {chunk.content}\n"
            context_parts.append(context)

        return "\n---\n".join(context_parts)

    def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> str:
        """Answer from retrieved chunks, without the citation contract.

        The plain path, used by every profile but ``hybrid_rrf``, which has
        its own labelled, budgeted, cited assembly in ``_answer_grounded``.
        """
        logger.debug(f"Generating answer for query: {query}")
        logger.debug(f"Number of retrieval results: {len(retrieval_results)}")

        if not retrieval_results:
            logger.warning("No retrieval results available for answer generation")
            return "I don't have enough information to answer this question. Please try rephrasing or ask something else."

        vectordb_name = self.vector_db.get_name() if hasattr(self.vector_db, 'get_name') else None
        embedding_model_name = self.embedding_model.get_name() if hasattr(self.embedding_model, 'get_name') else None
        RAGLogger.log_retrieval_results(
            logger,
            query,
            retrieval_results,
            vectordb_name=vectordb_name,
            embedding_model_name=embedding_model_name,
        )

        retrieved_context = self.get_retrieval_context(retrieval_results, include_metadata=True)
        RAGLogger.log_chunks_passed_to_llm(logger, retrieval_results, retrieved_context)

        system_prompt = """You are a helpful AI assistant. Your task is to answer the user's question based ONLY on the provided context from retrieved documents.

Guidelines:
- Use ONLY information from the provided context
- If the context doesn't contain enough information, say so clearly
- Cite the document name when using specific information
- Be concise but comprehensive
- If multiple documents provide relevant information, synthesize them
- Maintain a professional and helpful tone
- Respond in the same language as the user's question"""

        user_prompt = f"""Context from retrieved documents:
{retrieved_context}

Question: {query}

Please provide a clear and accurate answer based on the context provided above."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        logger.info("Calling LLM to generate answer...")
        RAGLogger.log_llm_request(logger, messages, temperature, max_tokens)

        try:
            with T.stage(T.ANSWER):
                response = self.answer_model.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            RAGLogger.log_llm_response(logger, response, success=True)
            logger.info(f"Answer generated successfully. Length: {len(response)} chars")
            return response.strip()
        except RESOURCE_CONTROL_EXCEPTIONS:
            # Not an answer: the query was stopped by its own limits, and
            # the route turns that into a truthful status rather than a
            # sentence that looks like a reply.
            raise
        except Exception as e:
            error_msg = f"I encountered an error while generating the answer: {str(e)}"
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            RAGLogger.log_llm_response(logger, str(e), success=False)
            return error_msg

    def query(
        self,
        question: str,
        top_k: int = None,
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Retrieve and answer: the whole question path, on one thread."""
        logger.info("=" * 80)
        logger.info(f"QUERY START: {len(question)} chars")
        logger.debug(f"Question: {question}")
        logger.info("=" * 80)
        started = time.perf_counter()

        logger.info("Phase 1: Document Retrieval")
        with T.stage(T.RETRIEVE):
            retrieval_results, metadata = self.retrieve(query=question, top_k=top_k)

        logger.info(f"Retrieved {len(retrieval_results)} documents")

        if self.retrieval_profile == 'hybrid_rrf':
            return self._answer_grounded(
                question, retrieval_results, metadata, temperature, max_tokens, started,
            )

        logger.info("Phase 2: Answer Generation")
        answer = self.generate_answer(
            query=question,
            retrieval_results=retrieval_results,
            temperature=temperature,
            max_tokens=max_tokens
        )

        sources = []
        for result in retrieval_results:
            sources.append({
                'document': result.chunk.doc_title,
                'doc_id': result.chunk.doc_id,
                'section': result.chunk.section_title,
                'pages': _source_pages(result.chunk.metadata),
                'score': result.score,
                'content_preview': result.chunk.content[:200] + '...' if len(result.chunk.content) > 200 else result.chunk.content
            })

        logger.info("QUERY COMPLETE")
        return {
            'answer': answer,
            'sources': sources,
            'metadata': metadata
        }
