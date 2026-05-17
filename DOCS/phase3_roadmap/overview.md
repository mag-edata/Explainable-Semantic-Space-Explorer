# Improvement Overview

**Created:** 2026-05-08
**Targets:** `core/`, `transforms/`, `ui/`, `data_pipeline/`, `tests/`, `DOCS/`, `data/manifest.json`

This document centrally manages the issues and tasks of the repository.

---

##  Priority Summary

| Priority | Item | Reason |
|----------|------|--------|
| **High** | 1.1 Risk of count mismatch between vocab and static_vectors | Regeneration after deployment may render the application unbootable |
| **High** | 3.1 Dependence on the current working directory at startup | Candidate blocker for Streamlit Cloud deployment |
| **High** | 3.4 Duplicate computation across tabs | May violate the requirement "within 5 seconds" |
| **High** | 9.2 `data/` large-file issue | Most critical blocker of the current phase (Streamlit Cloud deployment incomplete) |
| **Medium** | 1.2 Loss of context in POS tagging | Reliability of POS information for an explainability tool |
| **Medium** | 2.1 Word2Vec seed/workers reproducibility | Violation of CONST-06 |
| **Medium** | 1.4 Asymmetric normalization between static and contextual | Explainability on the UI becomes asymmetric |
| **Medium** | 2.2 UMAP has no automatic adjustment of n_neighbors | Warnings raised / visualization quality |
| **Medium** | 9.1 Altair shape legend inconsistency | Demo quality |
| **Low** | Others | Refactoring / code quality |

---

## On the File Split

| File | Contents |
|------|----------|
| `overview.md` (this file) | Title, metadata, priority summary |
| `high_priority.md` | 4 high-priority items (1.1 / 3.1 / 3.4 / 9.2) |
| `medium_priority.md` | 5 medium-priority items (1.2 / 1.4 / 2.1 / 2.2 / 9.1) |
| `low_priority.md` | 24 low-priority items (everything else) |
