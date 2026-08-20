# Frozen V4 equivalence fixtures

These fixtures are byte-for-byte copies or deterministic derivatives of
`erenayd58/chunk` commit
`1e7f7186c13729c739ccb3170da0892f7350cb27` (`phase5-holdout-validation`).

- `kkb-2024.units.jsonl` is the frozen canonical input.
- `authoritative-a4-chunks.jsonl` and `authoritative-a4-boundaries.jsonl` are
  the Phase-5 A4/V4 outputs.
- `boundary-embeddings.npz` consolidates the exact cached boundary vectors
  needed to run the pinned V4 core offline. It is not a new model or algorithm.
- `manifest.json` records and verifies every fixture SHA-256.

To rebuild only the compact embedding snapshot from a frozen checkout:

```powershell
py -3.11 tests/fixtures/frozen_v4/build_embedding_snapshot.py `
  --checkpoint-root ..\chunk `
  --output tests/fixtures/frozen_v4/boundary-embeddings.npz
```
