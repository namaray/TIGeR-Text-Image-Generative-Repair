# TIGeR — Code Fix Backlog

Working document. One entry per defect, ordered so that fixes which *gate the
measurement of other fixes* land first.

**Status:** `TODO` · `DOING` · `DONE` · `BLOCKED`
**Line references** are current as of branch `diagnostics/v2t-estimator-attribution`.

---

## Read this before starting

Two dependency traps:

1. **A4 must land before A1.** All five ablation configs share nested dicts via
   `cfg.copy()` (shallow). The moment A1 writes `cfg["arbiter"]["gamma"] = 0.0`,
   it mutates *every* config including Full System, and all five runs silently
   become γ=0. Fixing A1 first produces confidently wrong numbers.

2. **Section A gates Section B.** Every accuracy fix in B is measured by the
   ablation harness. If the harness is broken, you cannot tell which B fix
   helped. Land A first, re-run once to get a trustworthy baseline, then start B.

Expected sequence: `A4 → A1 → A2 → A3 → A5` → baseline run → `B0 (done) → B1…B6`.

---

## A. Measurement correctness — blocks everything else

### A1 · The "No Gamma Gate" ablation never disables the gamma gate
**Severity:** Critical — invalidates a published result
**Where:** `tiger/eval/repair_ablation.py:191` writes · `tiger/arbiter.py:197` reads

```python
cfg_no_gamma["fusion"]["gamma"] = 0.0     # written here
gamma = float(acfg.get("gamma", 0.60))    # read from cfg["arbiter"], not cfg["fusion"]
```

There is no `fusion:` section in `configs/tiger.yaml` at all, so the γ=0 run
executes the **identical configuration** as Full System. This is why the rows
matched (163/269/52.6% on Fashion, 35/444 on ABO) — it is the same run twice.

**Consequence:** `paper_assets/paper_draft_materials.md` §7.5 — the "cascading
safety net / early exit" analysis — explains a phenomenon that never occurred.
The gamma gate has still never been ablated. That section must be withdrawn or
rewritten after a corrected run.

**Fix:** write to `cfg_no_gamma["arbiter"]["gamma"] = 0.0`. **Requires A4 first.**

**Status:** TODO

---

### A2 · VLM judge points at a non-existent model, and the failure mode is a silent veto
**Severity:** Critical
**Where:** `tiger/vlm_judge.py:116`, retry list at `:201`, veto branch at `:208`

Default is `gemini-3.5-flash-lite`, which is not a real Gemini model ID (the
family runs 1.5 → 2.0 → 2.5 → 3). Several commits cycled through
`gemini-3.7-flash` and `gemini-3.1-pro`, which are also not real.

An invalid ID raises `404 NotFound`. That is not in the retry list
(`429/503/504`), so it falls to the fatal branch — which **returns `False`,
i.e. vetoes the repair**. A run with `--vlm-judge` would therefore veto 100% of
repairs while printing one error line per call.

**Fix:**
1. `genai.list_models()` with a live key to get valid IDs; pin a real one.
2. Add `404` / `NotFound` to a **fail-fast** branch — an invalid model is a
   configuration error and must raise at construction, not degrade to a veto.
3. Validate the model at `__init__` rather than on first call.

**Also determine:** whether any number in `paper_assets/` came from a
`--vlm-judge` run. If so it is unusable.

**Status:** TODO

---

### A3 · The random baseline is malformed
**Severity:** High — the 3.2% figure depends on it
**Where:** `tiger/eval/repair_ablation.py:27` · `tiger/arbiter.py:35`

`DummyArbiter` emits an `"E4"` key, but `CLASSES = ["E1","E2","E3","CLEAN"]`.
E4 is produced *by the gamma gate*, never predicted. When `E4` scores highest,
`route()` matches neither `CLEAN` nor `E1`, falls through to the final return,
and is labelled **E3/BOTH** — so ~25% of "random" routes are silently
mislabelled rather than random.

It also hardcodes `"CLEAN": 0.0`, so the baseline can never dismiss a row —
it is not a uniform random baseline over the real outcome space.

**Fix:** sample over the four real classes `["E1","E2","E3","CLEAN"]`; drop the
`E4` key; seed the RNG (see A3b).

**Status:** TODO

---

### A3b · `DummyArbiter` is unseeded
**Severity:** High — "No Arbiter" is non-reproducible between runs
**Where:** `tiger/eval/repair_ablation.py:30` (`import random` inside `predict_proba`)

Uses the global `random` module with no seed, so the random-routing row changes
every run and cannot be reproduced for the paper.

**Fix:** hold a `random.Random(seed)` instance on the class; take the seed from
config so it is recorded with the run.

**Status:** TODO

---

### A4 · `cfg.copy()` is shallow — the trap that breaks A1
**Severity:** High (blocking)
**Where:** `tiger/eval/repair_ablation.py:149, 158, 188`

All five ablation configs share the same nested dicts. Nothing leaks *today*
only because the gamma write creates a fresh `fusion` dict that nobody reads.
Correcting A1 without fixing this mutates every config at once.

**Fix:** `import copy` → `copy.deepcopy(cfg)` at all three sites.
**Land this before A1.**

**Status:** TODO

---

### A5 · "Restoration Accuracy" measures colour only, and never scores image repairs
**Severity:** High — the column name overstates what is measured
**Where:** `tiger/eval/repair_ablation.py:83`

```python
for _, r in audit[audit["field"] == "color"].iterrows():
```

Ground truth is built solely from colour rows. Consequences:
- V2T patches to `material` / `pattern` are committed but never scored.
- **T2V (image) repairs are never scored at all** — the ablation says nothing
  about image repair quality.
- The headline is therefore colour-patch accuracy on N=19, not "restoration".

**Fix:**
1. Build ground truth for every audited field, not just colour.
2. Add a separate T2V correctness metric (did the swapped image come from the
   originally-correct product?) — the audit log already records the swap.
3. Rename the column to what it measures, and report N per metric.

**Status:** TODO

---

## B. Repair accuracy — the actual goal

The architecture claims *"the image is ground truth; read the true value from
it."* In practice every V2T value comes from `solver._corrected_value`
(`tiger/solver.py:164`), which returns either the HSV histogram estimate or the
CLIP probe argmax. **Neither estimator localises the product.** Routing can be
perfect and the written value will still be wrong roughly half the time.

### B0 · Estimator attribution instrumentation
**Status:** ✅ **DONE** — commit `8ba8d19`

Records both estimator candidates on every repair plus per-estimator
counterfactuals, so operator error can be attributed to the pixel path or the
CLIP path. Emits `data/outputs/v2t_estimator_diagnostics.csv` and a printed
report. Chosen value is bit-identical to previous behaviour.

**Run this before B1–B6** — it tells you which estimator to fix first and the
ceiling any selection rule can reach.

---

### B1 · Skin is counted as product colour
**Severity:** High — systematic bias on fashion photography
**Where:** `tiger/colors.py:26` (`HUE_RANGES`), brown rule inside `_hue_to_name`

`orange` is hue 14–40°, and `brown` is hue < 50° with `v<0.6, s>0.2`. Human skin
sits squarely inside both. On short-sleeved, sleeveless, or full-body model
shots, exposed arms/face/legs can dominate the sampled region — biasing the
estimate toward orange/brown.

**Fix:** mask skin-tone pixels before histogramming (standard HSV skin range,
roughly hue 5–35° with bounded saturation/value), then renormalise. Cheap, and
expected to be one of the larger single wins on Myntra-style imagery.

**Status:** TODO

---

### B2 · No product localisation — the estimator measures the wrong object
**Severity:** High — affects 3 of 4 categories
**Where:** `tiger/colors.py:80` (fixed central 70% box)

The central box is the model's torso regardless of what the product is. For
**hats** (top of frame), **shoes** (bottom), and **bags** (held to the side),
the estimator is measuring a different object entirely.

**Fix, cheapest first:**
1. Category-conditioned crop regions (hats → upper third, shoes → lower third).
2. Background removal / saliency to isolate the foreground object.
3. CLIP/SigLIP patch-level attention to localise the described product.

**Status:** TODO

---

### B3 · The white-discount rule breaks on genuinely white products
**Severity:** Medium — fails an entire common colour class
**Where:** `tiger/colors.py:119`

White is discarded and the remainder renormalised whenever white is below 85%.
A genuinely white shirt at 84% therefore returns whatever shadow noise ranks
second. White is one of the most common fashion colours, so the threshold is
brittle at exactly the common case.

**Fix:** decide white-as-background from *spatial* evidence (is it connected to
the border?) rather than a global proportion threshold. Fall back to the
proportion rule only when the mask is unavailable.

**Status:** TODO

---

### B4 · `pixel_color_confidence` is not a confidence
**Severity:** Medium
**Where:** `tiger/colors.py:126` produces it; `tiger/solver.py:164` gates on `>= 0.55`

The value is the winning colour's **pixel share**. A 60% share of a badly-chosen
region is not a 60% probability of being correct, so the `0.55` gate is a magic
number applied to an uncalibrated quantity.

**Fix:** calibrate against actual correctness using the B0 diagnostics — fit
`P(correct | share, agreement, category)` on scored cases and gate on that.
Requires a B0 run first.

**Status:** BLOCKED on B0 run

---

### B5 · Aspect ratio is destroyed before cropping
**Severity:** Low
**Where:** `tiger/colors.py:76` — `resize((size, size))`

Squashes tall product images, shifting which body region lands in the centre
box and compounding B2.

**Fix:** resize preserving aspect ratio, then crop.

**Status:** TODO

---

### B6 · The two estimators never cross-check
**Severity:** High — this is the highest-value *architectural* fix
**Where:** `tiger/solver.py:164`

The logic is `if/else`: a confident-but-wrong pixel estimate silently wins and
the CLIP probe is never consulted. **Disagreement between the two is exactly the
signal worth acting on**, and it is currently discarded.

**Fix:** agree → repair; disagree → escalate. This converts silent wrong-writes
into escalations, raising restoration accuracy on the acted-on set by trading
coverage. Report as a **risk–coverage curve**, not a point — that is the
standard form in the selective-prediction literature and makes the trade
explicit rather than looking like threshold tuning.

B0's report already quantifies what this would buy before it is built.

**Status:** TODO (size it from the B0 run)

---

### B7 · CLIP is the weakest available encoder for attribute binding
**Severity:** Medium — sets the ceiling on the probe path
**Where:** `configs/tiger.yaml:12`, `compare_encoders` at `:17`

ARO (ICLR 2023) measures CLIP at **62%** on attribution where chance is 50%,
against BLIP 88% and XVLM 87%. The pipeline uses CLIP for an attribute-centric
task. The probe implementation itself is sound (category-conditioned,
prompt-ensembled, normalised at `tiger/sieve.py:128`) — the encoder is the limit.

**Fix options:**
1. Swap the probe encoder to BLIP/XVLM — `compare_encoders` already supports it.
2. Apply the Koishigarina et al. (ICLR 2026) linear transform on text
   embeddings, which recovers cross-modal binding **from the existing embedding
   cache** with no re-encoding and no retraining.

**Status:** TODO

---

## C. Configuration & reproducibility

### C1 · γ has four different values across the repo
**Severity:** Medium — nobody can say which threshold produced which result
**Where:** `configs/tiger.yaml:67`

| Source | Value |
|---|---|
| `configs/tiger.yaml:67` | **0.40** |
| ABO analysis + `honest_limitations.md` | "default **0.60**" |
| `paper_assets/paper_concepts.md` | **0.85** |
| ABO recalibrated | **0.448** |

The ABO "75.2% fall below the default threshold" finding depends entirely on
0.60 being the value that actually ran. Reading the committed confidence
histogram, at γ=0.40 only ~5% of items fall below — not 75%.

**Fix:** establish which value the ABO run used, correct every document to
match, and record the effective γ in the run output so this cannot recur.

**Status:** TODO

---

### C2 · No `--gamma` CLI flag
**Severity:** Medium
**Where:** `tiger/cli.py` (absent)

Recalibration requires editing `configs/tiger.yaml` in place, which is how C1
arose. Their own note flags the risk of leaving the config wrong between the
fashion and ABO runs.

**Fix:** add `--gamma` to `repair` and `ablate-repair`, overriding config; echo
the effective value into the run output.

**Status:** TODO

---

### C3 · Tests deleted but still declared
**Severity:** Medium — reproducibility claim for an applied-venue paper
**Where:** `tests/` (only stale `__pycache__` remains) · `pyproject.toml:41`

`testpaths = ["tests"]` and the `test` extra still ship, so `pytest` collects
nothing. The README previously advertised 68 unit tests pinning the critical-
review fixes (F1/F3/F6/F10/F12, the Eq. 27–29 gates, routing constraints,
fusion quarantine) — that was a genuine credibility asset.

**Fix:** restore the suite from git history (`git show <pre-strip>:tests/...`),
or remove the pytest config and drop the claim. Do not leave it declared-but-empty.

**Status:** TODO

---

### C4 · Hardcoded generated-image path
**Severity:** Low
**Where:** `tiger/solver.py:234,236`

`data/sample/images/generated/` regardless of dataset, so ABO and Fashion
artifacts are written into the synthetic sample tree.

**Fix:** derive from `cfg["data"]` with a per-run subdirectory.

**Status:** TODO

---

### C5 · Evaluation artifacts exist nowhere in the repo
**Severity:** Medium
**Where:** `data/outputs/` is gitignored and empty

Every number in `paper_assets/` traces to Kaggle CSVs that survive in no
committed form. For a systems paper this is the reproducibility surface.

**Fix:** commit the summary CSVs (not the caches) under `paper_assets/results/`.

**Status:** TODO

---

## D. Robustness & design

### D1 · `class_weight="balanced"` undercuts the calibration claim
**Severity:** Medium
**Where:** `tiger/arbiter.py:129`

Class weighting distorts probability calibration, yet the γ-gate depends on
calibrated `predict_proba` and `reviewer_defense.md` Attack 8 claims logistic
regression was chosen *because* it is well-calibrated.

**Fix:** produce a reliability curve (`arbiter.reliability_table` already
exists). If calibration is poor, add Platt/isotonic recalibration on a holdout,
or drop the balancing and handle imbalance in the threshold instead.

**Status:** TODO

---

### D2 · Asymmetric failure handling in the independent verifier
**Severity:** Low — intentional, but undocumented in the paper
**Where:** `tiger/verify.py:161` (`check_v2t` returns `True` on unreadable image)
vs `check_t2v` (returns `False` on unreadable candidate)

Deliberate and commented: do not veto on our own read failure, but do not trust
an unverifiable candidate. Sound, and should be stated explicitly rather than
discovered by a reviewer.

**Status:** TODO (documentation)

---

### D3 · Position bias in `check_t2v`
**Severity:** Low
**Where:** `tiger/verify.py` — old/new images presented in fixed order

MLLM-as-a-Judge (ICML 2024) names position bias as a failure mode present even
in GPT-4V. The comparative check is otherwise the *reliable* setting for a VLM
judge, so this is worth controlling.

**Fix:** evaluate both orderings and require consistency; count disagreement as
a veto.

**Status:** TODO

---

## E. Documentation contradicted by the code

Fix **after** the corrected ablation run — the numbers will move.

### E1 · `reviewer_defense.md` Attack 8: "exactly four continuous evidence metrics"
The `FEATURES` list in `tiger/arbiter.py` has **14**. The rebuttal also argues
from "a 4-dimensional input space". Trivially checkable by any reviewer.
**Status:** TODO

### E2 · `reviewer_defense.md` Attack 3: the 47.4% were *not* "safely escalated"
52.6% is restoration accuracy among the **163 repaired** rows; escalated rows
are the separate 269. Those ~9 cases were **committed with wrong values**. The
rebuttal presents a real error rate as a safety guarantee — the most dangerous
line in that document.
**Status:** TODO

### E3 · "Perfectly overlapped" was never demonstrated
Identical aggregate counts do not prove identical *sets*; two different sets of
444 items produce the same total. Given A1, the claim is moot — but if the
early-exit story survives a corrected run, prove it by intersecting the
escalated `row_id` sets.
**Status:** BLOCKED on A1

### E4 · Attack 1 argues cost where the durable argument is reliability
The "why not just use a VLM" rebuttal rests on cost, throughput, and rate
limits. Cost arguments age badly. MLLM-as-a-Judge (ICML 2024) shows judges
diverge from humans on *absolute scoring* and exhibit position, egocentric, and
length bias plus hallucination even in GPT-4V — a reliability argument that does
not expire.
**Status:** TODO

### E5 · Attack 6 does not answer the ARO objection
The model-agnostic defence answers "why not a newer model" but not "why the
model measured worst at your exact task" (CLIP 62% vs BLIP 88% on attribution).
Pairs with B7.
**Status:** TODO

---

## Summary

| Section | Items | Blocking? |
|---|---|---|
| A. Measurement correctness | 6 | Yes — gates all of B |
| B. Repair accuracy | 8 (1 done) | The actual goal |
| C. Config & reproducibility | 5 | Partly |
| D. Robustness & design | 3 | No |
| E. Documentation | 5 | After A |

**Next action:** `A4` (deepcopy) → `A1` (gamma wiring) → re-run `ablate-repair`
with B0 instrumentation active → read the estimator attribution report → that
report decides whether B1/B2 (pixel path) or B7 (encoder path) comes first.
