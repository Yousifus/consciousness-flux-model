# Multi‑Agent Handoff Ledger (HANDOFF.md)

A single source of truth for moving intent and context across **Yousef ↔ Glyph ↔ Perplexity ↔ Cursor**.  
Keep this file short, living, and always updated at the **top** with the current Context Packet.

---

## 1) Current Context Packet (fill & update each cycle)

```yaml
intent: "Rich features integration COMPLETE - ready for real telemetry integration phase."
repo: "F:\\Projects\\Consciousness-flux-model"
github: "https://github.com/Yousifus/consciousness-flux-model"
branch: "main"
commit_ref: "ab80998"   # Rich features implementation commit
timestamp: "2025-09-13T12:30:00+00:00"
artifacts_of_record:
  - "outputs/results/model_results_*.json"      # Enhanced with hierarchy_weights_sample, changepoints
  - "outputs/images/consciousness_flux_analysis_*.png"
  - "outputs/results/csr_by_priors.csv"         # when compare_priors is run
  - "outputs/images/csr_by_priors.png"
  - "outputs/results/regional_panel.csv"        # when --regions 3 is enabled
  - "src/rich/*.py"                              # New rich feature modules
constraints:
  - "No new deps beyond numpy/pandas/matplotlib/pytest - MAINTAINED"
  - "Preserve CLI defaults and backward compatibility - VERIFIED"
  - "Tests must remain green (pytest -q) - PASSING"
acceptance_tests:
  - "✅ CSR recomputed from fitted params (not synthetic columns)"
  - "✅ Supply plot shows terrestrial & emergent in CCU/yr; no unit mismatches"
  - "✅ results JSON includes priors, weights_used, fit_source, regime_year, artifact_image"
  - "✅ Network effect used in only one direction (demand efficiency OR supply gate)"
  - "✅ Rich features: --rich, --quality, --changepoints, --regions all functional"
  - "✅ Quality multiplier (0.85-1.25) applies to supply when --quality enabled"
  - "✅ Changepoint detection finds regime transitions in log-CSR"
  - "✅ Regional panel generates 3-region split when --regions 3"
rich_features_added:
  - "Correlated drivers: tech adoption curves, cultural quality proxies, mystical shocks"
  - "Quality multiplier: bounded supply enhancement based on education/philosophy/meditation/psychedelics"
  - "Hierarchy weights: multi-scale consciousness levels (quantum→cosmic) with temporal evolution"
  - "Changepoint detection: binary segmentation on log-CSR for regime discovery"
  - "Regional panels: synthetic 3-region split (DEV/DEVG/FRONT) with connectivity variations"
open_questions:
  - "Real telemetry integration: C (compute FLOPs/yr), U (creators/users), L (connectivity) datasets since 1990"
  - "Rich driver calibration: map synthetic correlations to real tech/culture time series"
  - "Quality governance: external validation sources for consciousness quality multipliers"
  - "Regional boundaries: geographic vs socioeconomic splits for multi-region analysis"
due_to_next_agent: "Perplexity → Research phase: Find 3 candidate datasets each for C, U, L + cultural quality proxies (education, philosophy publications, meditation/mindfulness adoption, psychedelic research trends) with coverage 1990-2025."
```

---

## 2) Roles (fixed)

- **Yousef (Product Owner):** sets intent, constraints, and acceptance tests; merges decisions.  
- **Glyph (Architect/Integrator):** formalizes math & specs; verifies outputs; writes Cursor-ready diffs.  
- **Perplexity (Research Scout):** finds sources, compares claims, returns a concise, cited brief.  
- **Cursor (Builder):** applies file edits locally, runs commands, returns git diff + console summary.

**Source of truth:** this repo (manifest + results).

---

## 3) Message Templates (copy, fill, paste)

### A) To Perplexity — Research Brief
```
ROLE: Research Scout
TASK: Find and compare the BEST primary datasets on compute FLOPs/yr (C), creators/users (U), and connectivity per-capita (L) since 1990.
CONTEXT: <paste Current Context Packet>
DELIVERABLE:
- 5 concise bullets of findings with dates
- 3 disagreements/tensions across sources
- 3 candidate series per driver (name, provider, cadence, coverage) with 1‑line why each matters
- 5 citations (URLs)
BOUNDARIES: Prefer official or widely-used datasets; no speculation.
```

### B) To Glyph — Model/Plan Spec
```
ROLE: Architect/Integrator
TASK: Translate the research brief into equations + file changes.
CONTEXT: <Current Context Packet> + (Perplexity brief)
DELIVERABLE:
- Equations/algorithms (units explicit) and any new caps/gates
- File-level diff plan (paths, functions, signatures)
- Tests to add (names + assertions)
- Expected artifacts & metric shifts
```

### C) To Cursor — Patch Request
```
ROLE: Builder
TASK: Apply code changes locally.
CONTEXT: <Current Context Packet> + (Glyph spec)
ACTIONS:
- Edit listed files & functions
- Add tests as specified
- Run:
    cd src && python consciousness_flux_model_v1.py --priors IIT
    python src/compare_priors.py
    pytest -q
ACCEPTANCE:
- All tests green
- New JSON/PNG artifacts present
- Diff matches the plan
OUTPUT: Paste git diff + console output summary.
```

### D) To Claude — Critical Review
```
ROLE: Reviewer/Challenger
TASK: Red‑team the change.
CONTEXT: <Current Context Packet> + (Cursor diff + results JSON)
DELIVERABLE:
- 5 potential failure modes
- 3 sensitivity checks to run next
- 1 simplification and 1 generalization
- Mark any claim with >10% fragility as ⚠️
```

---

## 4) Operating Loop (MAP)

1) **Intent** (Yousef) → write/update the Context Packet.  
2) **Research** (Perplexity) → return a Research Brief.  
3) **Design** (Glyph) → return a Model/Plan Spec.  
4) **Build** (Cursor) → apply patches + run.  
5) **Verify** (Glyph) → check artifacts vs. acceptance tests.  
6) **Decide** (Yousef) → approve or iterate.  
7) **Record** (All) → update manifest, results, and this log.

Keep cycles small (≤ 90 minutes) and shippable.

---

## 5) Run Log (append at top)

**2025‑09‑13T12:30 — Cycle B (Rich Features Integration)**  
- **Intent:** Add rich synthetic data features: correlated drivers, quality multiplier, hierarchy, changepoints, regional panels  
- **Spec:** Implement src/rich/ modules; wire CLI flags (--rich, --quality, --changepoints, --regions); preserve v1.0.3 defaults  
- **Patch:** 5 new rich modules, enhanced main model with optional features, updated tests, CLI integration  
- **Artifacts:** `outputs/results/regional_panel.csv`, enhanced JSON with hierarchy_weights_sample & changepoints  
- **Verdict:** ✅ All features functional, tests passing, backward compatibility maintained  
- **Notes:** Quality multiplier (0.85-1.25), changepoint detection finds 7 regime transitions, 3-region panel ready

**2025‑09‑13T09:27 — Cycle A (Stabilization)**  
- **Intent:** stabilize v1.0.3; add priors compare; wire provenance.  
- **Patch:** supply units aligned, fitted CSR recompute, seeded bands, priors CLI, provenance fields.  
- **Artifacts:** `model_results_v1.0.3_*.json`, `consciousness_flux_analysis_v1.0.3_*.png`.  
- **Verdict:** ✅

---

## 6) Invariants (tests should guard these)

- Positivity: (S>0, D>0, CSR ≥ 0)  
- Single-job network effects: demand uses e_share **or** supply uses sigma_L, not both  
- Units: supply plotted in **CCU/yr** for terrestrial & emergent  
- Fitting: uses model slice; results aren’t copied from synthetic columns  
- Provenance: results JSON links the figure; includes priors, weights_used, fit_source, regime_year

---

## 7) Glossary

- **CSR:** Consciousness Surplus Ratio = S/D  
- **Terrestrial:** biological contributions (animals/plants) in CCU/yr  
- **Emergent:** alpha * C^beta * U^gamma * sigma_L * cap(C,U) in CCU/yr  
- **Context Packet:** minimal header that keeps agents aligned

---

*Last updated: 2025-09-13T12:30:00+00:00*
