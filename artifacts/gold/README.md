# Frozen retrieval gold sets

A file here is a **frozen** regression set: a list of questions with the source
a human confirmed answers each one. `python -m cli eval` reads one of these and
recomputes every rank from scratch.

These are inputs to evaluation, so they are tracked in git and are not edited
by the app.

`.gold_set.json` in the repository root is the opposite: mutable runtime state
holding whatever has been marked in the review screen so far, including trial
and mistaken marks. It is gitignored. Freeze a reviewed subset with:

```
python -m cli gold export --kb <name-or-id> --out artifacts/gold/<name>.json --all
```

Export never treats a runtime entry as verified on its own; read what it prints
and drop anything that should not be part of the regression set.
