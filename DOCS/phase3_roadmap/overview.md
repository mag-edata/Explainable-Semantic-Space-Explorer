# Improvement Overview

**Created:** 2026-05-08
**Updated:** 2026-07-03
**Targets:** `core/`, `transforms/`, `ui/`, `data_pipeline/`, `tests/`, `DOCS/`, `data/manifest.json`

This document centrally manages the issues and tasks of the repository.

---

## Phase-3 Policy (fixed 2026-07-03)

On 2026-07-03 the project purpose was redefined (requirements definition v2.0): the portfolio purpose was retired and the system is now developed as a real product for **NLP learners**. Work proceeds in stages; the items in this roadmap map onto the stages as follows.

| Stage | Contents | Related roadmap items |
|-------|----------|-----------------------|
| **Phase A — data quality & UX repair** | Vocabulary curation + full asset regeneration (extends 8.5; resolves 1.1 as a by-product), context-valid POS regeneration (1.2), Word2Vec seed fix (2.1), normalization unification (1.4), cross-tab caching + form-based execution (3.4, 8.4), guided UX (new: example-query buttons, OOV suggestions, plain-language verdicts). *Conditional:* corpus expansion (e.g., WikiText-103) only if neighborhood quality remains insufficient after curation | 8.5, 1.1, 1.2, 2.1, 1.4, 3.4, 8.4 |
| **Phase B — sentence-context mode** | In-context token vector extraction, two-sentence comparison for sense separation, comparison-first screen restructure | New — see requirements v2.0 FR-20–22, FR-27 |
| **Phase C — release** | Streamlit Cloud deployment (3.1, 9.2; assets shrink after Phase A), user-facing README rewrite | 3.1, 9.2 |

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
