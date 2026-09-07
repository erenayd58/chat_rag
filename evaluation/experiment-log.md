# Retrieval experiment log

One row per experiment, in the order they were run. An experiment changes one
thing, is measured on three levels, and is either adopted as the new baseline
or recorded and reverted. Numbers come from the production HTTP path
(`/api/query`) for the SBB dev set and from the real `assemble_context` over
frozen retrieval for the KKB gold set.

**Levels**

- **A — SBB dev set** (10 questions, `Strateji-ve-Butce-Baskanligi-2024`, KB
  `deneme_v4`, 175 Deep chunks). Gold is at chunk level; answers are judged
  against fact markers and the "kaynaklarda yeterli bilgi yok" sentence.
- **B — KKB-2024 gold set** (47 questions, unit-level evidence,
  `evaluation/kkb-2024/retrieval-benchmark/canonical-v3/gold-queries-v2`).
  Retrieval frozen per configuration, no answer model: this level measures
  context construction, not generation.
- **Q7 sentinel** — the one dev question where neighbour expansion was
  observed to earn its keep (`d-chunk-0022` reached the context only as a
  neighbour). Watched separately in every experiment.

`irrelevant token ratio` follows `amsc.research.benchmark.retrieval._irrelevant_token_ratio`,
applied to the assembled context rather than to top-K.

**Decisions**

| experiment | decision | what is in the code now |
|---|---|---|
| E1 — hits first, expansion on the remainder | **ACCEPTED** | shipped in `components/context/assembler.py`, with `dropped_ranked_hits` in the query trace |
| E2 — top_k 5 → 10 | **REJECTED** | reverted; `DEFAULT_TOP_K` stays at the E1 baseline of 5 |
| E3 — cross-encoder rerank over the fused pool | **REJECTED** | reverted; no reranker in the `hybrid_rrf` path, no `RERANK_*` settings, `CROSS_ENCODER_MODEL` back to `ms-marco-MiniLM-L-6-v2` |

A rejected experiment keeps its row below as a measurement, not as a plan:
the numbers stand, the code does not.

---

## Baseline (before E1)

| level | evidence-in-context | answer | dropped ranked hits | neighbour tokens |
|---|---|---|---|---|
| A | 7/10 | 7 correct, 2 false-insufficient, 1 wrong | 20/50 (40%) | 48% of context |
| B | 40/47 | not run | 56/235 (24%) | 47% of context |

Failure mechanism: `assemble_context` admitted each hit's neighbours before
the next ranked hit, so a hit at rank 5 could be dropped for a neighbour of
rank 1. On the dev set that produced two false "no information" answers and
one confidently wrong answer.

## E1 — hits first, expansion on the remainder — **ACCEPTED**

Two passes in `components/context/assembler.py`: every ranked hit is offered
the budget in rank order, then neighbours spend what is left. Reading order is
unchanged (seed then its neighbours). `ContextBundle.dropped_ranked_hits`
added so a dropped hit is countable apart from an unaffordable neighbour.

| level | evidence-in-context | answer | dropped ranked hits | neighbour tokens | irrelevant | citation precision |
|---|---|---|---|---|---|---|
| A | 7/10 | **8 correct**, 2 false-insuff, **0 wrong** | **2/50 (4%)** | **7%** | 0.860 | 0.944 |
| B | **43/47** | not run | **5/235 (2%)** | **20%** | 0.952 | n/a |

- Q8 (DEA) wrong → correct: its gold chunk was at rank 5 and had been dropped.
- B: 3 questions gained evidence, 0 lost.
- Control held: `evidence-in-ranked` unchanged at 42/47, as retrieval was untouched.
- Cost: expansion shrank from 47% to 20% of the budget. Q7 lost `d-chunk-0022`
  (2/2 → 1/2 evidence) but still answered correctly from another chunk.

## E2 — top_k 5 → 10 — **REJECTED**

| level | evidence-in-context | answer | dropped ranked hits | neighbour tokens | irrelevant | citation precision |
|---|---|---|---|---|---|---|
| A | 8/10 | **9 correct**, 1 false-insuff | 43/100 (43%) | 1% | 0.855 | 0.950 |
| B | 43/47 (unchanged) | not run | 158/470 (34%) | **1%** | 0.955 | n/a |

Reverted despite the dev-set gain (Q2 fixed). Reasons:

- B did not move: gold entered the ranked list for 2 more questions but only
  one reached the context, and one question that had been rescued by neighbour
  expansion lost it. A +1/-1 swap, not a gain.
- Dropped ranked hits rose to 34%; neighbour expansion effectively disappeared.
- Context size barely moved (3003 → 3050 tokens, 6.47 → 6.77 sources per
  query), which is the finding: **at top_k=5 the 3200-token budget, not K, is
  already the binding constraint.** Raising K buys ranked candidates that
  cannot be afforded.
- Q3 stayed unfixed, showing K alone is not the lever.

Reverted in code: `top_k` is read from the request or falls back to 5.

## E3 — multilingual cross-encoder over the fused pool — **REJECTED**

`RERANK_ENABLED=true`, pool 50, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
(mMARCO multilingual MiniLM-L12; the English `ms-marco-MiniLM-L-6-v2` it
replaces cannot score Turkish). `BAAI/bge-reranker-v2-m3` was the first choice
and was abandoned here: 2.3 GB over a link running at ~13 MB/min, and an
XLM-R-large encoder on CPU-only torch would multiply an already heavy latency.

| level | evidence-in-context | answer | dropped ranked hits | neighbour tokens | irrelevant | citation precision |
|---|---|---|---|---|---|---|
| A | **9/10** | **9 correct**, 1 false-insuff | 2/50 (4%) | 3% | **0.833** | **0.950** |
| B | 43/47 | not run | 7/235 (3%) | 12% | 0.953 | n/a |

Gold rank inside the 50-candidate pool, RRF → reranked:

| question | gold chunk | RRF | reranked |
|---|---|---|---|
| Q2 | d-chunk-0032 | 9 | **1** |
| Q7 | d-chunk-0022 | 36 | **4** |
| Q8 | d-chunk-0024 | 5 | **1** |
| Q3 | d-chunk-0031 | 6 | **14** |
| Q5 | d-chunk-0021 | 1 | 2 |

On B: gold rank improved on 10 questions, unchanged on 30, worsened on 3, and
fell out of the top 5 entirely on 4. Coverage 0.915 → 0.897.

Latency (CPU, pool of 50): rerank p50 **16-17 s**, end-to-end p50 **8.7 s → 25.8 s**.

Reading: strong on the dev set and on the sentinel, neutral-to-slightly-negative
on the regression set, and not shippable at this latency on this hardware. The
adopt/reject rule from the plan -- a change is accepted only when the held-out
level moves the same way -- is not met.

Rejected and reverted: the opt-in branch in `RAGPipeline`, the rerank call
path and its trace fields, the `RERANK_ENABLED` / `RERANK_POOL` settings and
the `.env` block are all gone; `hybrid_rrf` is a deterministic retrieve-and-
answer path again. Re-running E3 means re-applying that patch, the multilingual
cross-encoder and `HF_HUB_DISABLE_XET=1`, on hardware where 16-17 s of rerank
latency is not what ships.

---

## Standing findings

- The generation context budget (3200 tokens, ~5-6 chunks at this chunk size)
  is the binding constraint on how much retrieved evidence reaches the answer
  model. Both E2 and E3 ran into it from different directions.
- Q3 has resisted every experiment so far. It needs two facts from two chunks;
  the second (`d-chunk-0031`, a personnel table whose gender chart sits at the
  end) is ranked low by fusion *and* pushed lower by the cross-encoder.
- The Hub's Xet transfer path is blocked on this machine. `HF_HUB_DISABLE_XET=1`
  is set in `.env`; without it a cross-encoder load hangs on a revision check
  instead of using the cache it already has.
