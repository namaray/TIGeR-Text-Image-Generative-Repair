# TIGeR — Code Fix Backlog

Working document. One entry per defect, ordered so that fixes which *gate the
measurement of other fixes* land first.

**Status:** `TODO` · `DOING` · `DONE` · `BLOCKED` · `PARKED` · `WITHDRAWN`
**Marker:** ⚑ = changes the design, not just the numbers (see the ⚑ table below)
**Line references** are current as of branch `docs/fixes-backlog-audit`.
**Verified:** all 50 entries swept against the code on 2026-09-08. 41 held as
written; D3 withdrawn; B1/B2/B5 re-classified BLOCKED (no data in-repo can
exercise them); A2/A3/C3/C5/D12 corrected. Findings are recorded in-entry.

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

3. **D5 must land before D6 and D7.** Those two are the same normalisation
   defect, and both exist because D5 put five colours into the domain *and* into
   the alias map. Fixing the comparisons without fixing the domain leaves the
   probe candidates colliding.

Expected sequence: `A4 → A1 → A6 → A2 → A3 → A5`, plus `D4` and `D5 → D6/D7`
→ baseline run → `B0 (done) → B1…B6`.

`D4` and `D5` join the pre-baseline set because they change **pipeline
behaviour**, not just measurement: the numbers taken before them are not the
numbers the fixed system produces.

---

## ⚑ Three findings that change the design, not the numbers

Everything else in this file is a defect against a sound design — fix it and the
architecture stands. These three say the built system is not the described
system. Each needs a **decision** before it needs a patch.

The sweep split these into two kinds, which the first draft conflated.

**Changes the design — new behaviour that has never existed. `PARKED`: needs a
decision, not a patch. Do not fix these in a cleanup pass.**

| ⚑ | Finding | Entry | Why it is a design change |
|---|---|---|---|
| **1** | The repair operator has no ground-truth source, and nothing abstains on *value* | `B6` (with `B1`–`B4`, `E2`, `D6`) | The γ-gate abstains on routing; Eq. 27–29 abstains on schema and similarity. Nothing abstains on *"I do not know what colour this is."* A wrong-but-in-domain value that raises CLIP similarity passes every gate and is committed — the ~9 of 19 in `E2`. Fixing it inserts an abstention stage between Solver and Verify: a new component. |
| **2** | The closed loop is not closed, and E3 has no behaviour of its own | `D4` | Four taxonomy classes, three implemented behaviours. Either remedy — rows re-entering a pass, or a two-step `BOTH` plan — adds pipeline behaviour that has never run. |

**Changes a claim's scope — the architecture is untouched.**

| ⚑ | Finding | Entry | Why it is not a design change |
|---|---|---|---|
| **3** | Cross-domain generalisation was never tested for image repair | `A6` | The T2V policy gate is *designed* to be configurable. Widening `allowed_categories` is a one-line config edit. What moves is what RQ3 may claim, not the pipeline's shape. |

**Everything else in this file is architecture-preserving.** Sorted by blast
radius: docs/hygiene only (E1, E2, E4–E7, E10, E12, C6, C7, D2, D13); measurement
only (A1, A3, A3b, A4, A5, A8, C2, C3, C5, C8, E8); behaviour moves but the design
is intact (A2, A6, A7, C4, D5–D12, B1–B5, B7).

**Framing consequence.** What exists today is a *high-recall cross-modal error
detector with a calibrated triage layer, and a repair operator that is the
weakest component in the system.* That is a defensible paper on the committed
artifacts. The repair paper needs `B1`/`B2`/`B6` first. `reviewer_defense.md`
Attack 3 already reaches for the triage framing — it should be the accurate
description, not the fallback argument.

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

**Status:** DONE — `2d29713`; regression tests pin both directions.

---

### A2 · VLM judge points at a non-existent model, and the failure mode is a silent veto
**Severity:** Critical
**Where:** `tiger/vlm_judge.py:116`, retry list at `:201`, veto branch at `:208-210`
**Verified:** confirmed — `404` matches none of `429/503/504`, so an invalid model
ID reaches `return False` at `:210`.

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

**Status:** DOING — `53736e6` makes a misconfigured judge raise instead of veto. Pinning a verified model ID still needs a live key.

---

### A3 · The random baseline is malformed
**Severity:** High — the 3.2% figure depends on it
**Where:** `tiger/eval/repair_ablation.py:27` · `tiger/arbiter.py:35`

`DummyArbiter` emits an `"E4"` key, but `CLASSES = ["E1","E2","E3","CLEAN"]`.
E4 is produced *by the gamma gate*, never predicted. When `E4` scores highest,
`route()` matches neither `CLEAN` nor `E1`, falls through to the final return,
and is labelled **E3/BOTH** — the final `return` hardcodes `"E3"` regardless of
what `top` was.

**Verified, with one correction:** the mechanism is confirmed, but "~25%" is
loose. Four normalised uniforms rarely produce a max ≥ γ, so the gamma gate
intercepts many rows *before* the E4 fallthrough is reached. The share of routes
affected depends on γ and cannot be stated without a run.

It also hardcodes `"CLEAN": 0.0`, so the baseline can never dismiss a row —
it is not a uniform random baseline over the real outcome space.

**Fix:** sample over the four real classes `["E1","E2","E3","CLEAN"]`; drop the
`E4` key; seed the RNG (see A3b).

**Status:** DONE — `01c469a`.

---

### A3b · `DummyArbiter` is unseeded
**Severity:** High — "No Arbiter" is non-reproducible between runs
**Where:** `tiger/eval/repair_ablation.py:30` (`import random` inside `predict_proba`)

Uses the global `random` module with no seed, so the random-routing row changes
every run and cannot be reproduced for the paper.

**Fix:** hold a `random.Random(seed)` instance on the class; take the seed from
config so it is recorded with the run.

**Status:** DONE — `01c469a`; seed from `cfg["eval"]["random_baseline_seed"]`.

---

### A4 · `cfg.copy()` is shallow — the trap that breaks A1
**Severity:** High (blocking)
**Where:** `tiger/eval/repair_ablation.py:149, 158, 188`

All five ablation configs share the same nested dicts. Nothing leaks *today*
only because the gamma write creates a fresh `fusion` dict that nobody reads.
Correcting A1 without fixing this mutates every config at once.

**Fix:** `import copy` → `copy.deepcopy(cfg)` at all three sites.
**Land this before A1.**

**Status:** DONE — `d6c4936`, landed before A1 as required.

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

**Sweep finding — step 2 as written is unachievable.** `noise.swap_image`
COPIES a donor path over the row's own (`tiger/data/noise.py:158-178`), leaving
the row's true original referenced by no row; `CandidatePool` holds in-use paths
only (`tiger/solver.py:135-141`). The original is absent from the pool *by
design* — that is the F14 held-out protocol applied to the operator. Scoring
recovery of it would report 0% forever.

Implemented instead: a T2V repair counts as restored when the image it installs
depicts a product matching this row's true category and colour
(`image_provenance`). Generated images depict no catalogue product and count as
attempts but never successes — the conservative reading. Steps 1 and 3 landed
as specified.

**Status:** DONE — `804ed68`, with one deviation; see the sweep note above.

---

### A6 ⚑ · The T2V policy allowlist disabled image repair for the whole ABO run
**Severity:** Critical — a cross-domain claim rests on a path that never executed
**Where:** `tiger/arbiter.py:242` reads · `configs/tiger.yaml:73` sets

```python
cat_ok = ev.get("category") in (policy.get("allowed_categories") or [ev.get("category")])
if "T2V" not in allowed_dirs or not cat_ok:
    return Route(..., "HUMAN", "human_review", 3, f"{top} but T2V blocked by policy/modality")
```

`allowed_categories` is `["shirts","shoes","bags","hats"]`. ABO imports map to
`electronics / furniture / kitchen / home_decor` (`tiger/data/import_abo.py:33`).
Every ABO E2 and E3 row was force-escalated by a fashion-only allowlist **before**
schema validation or the gamma gate applied.

**Consequence:** RQ3 — "does TIGeR generalise to a new vertical via lightweight
schema adaptation?" — is unanswered for image repair. H10 blames the escalation
rate on the `color` requirement and H11 on Arbiter underconfidence; this is a
third cause, upstream of both, and unacknowledged in every document. It also
interacts with A1: on ABO, correcting the gamma wiring still cannot move E2/E3
outcomes while this gate is closed.

**Fix:** extend `allowed_categories` to the ABO verticals, or make the policy
schema-driven per domain; echo the effective policy into the run output.
**Land with A1, before the corrected ABO re-run** — otherwise the new numbers get
read the same wrong way.

**Status:** DONE — `a989d10`.

---

### A7 · Precision-floor fusion is never loaded by the live pipeline
**Severity:** High — the headline precision is not the operating point
**Where:** `tiger/cli.py:195` (`cmd_detect`) · `tiger/repair.py:74` (`run_repair_cycle`)

Both call `sieve_mod.apply_thresholds(sig, thr)` with no `fusion=` argument.
`data/thresholds/tiger_fusion.json` is read by `cmd_ablate` and nothing else.

**Consequence:** the advertised **P=0.888 / R=0.882 / F1=0.885** is an offline
ablation row. `detect`, `analyze`, `route`, `repair` and every repair-side number
run on the un-fused detector at **P=0.793 / R=0.924**. `tiger_project_doc.md` §4
presents the fused figure as the Sieve's output.

**Fix:** load the fusion config in `cmd_detect` and `run_repair_cycle` when it
exists, behind an explicit flag, and state per reported number which operating
point produced it. Note fusion trades recall for precision (mutate_text recall
0.853 → 0.773), so enabling it changes what reaches the repair stage.

**Status:** DONE — `2d0690c`; opt-in via `--fusion`.

---

### A8 · The per-signal precision floor measures row dirtiness, not signal correctness
**Severity:** Medium — the 0.85 floor is weaker than it reads
**Where:** `tiger/fusion.py:72`

```python
prec = float(dirty[fired].mean())
```

A row counts as a true positive for a probe whenever the row is dirty **for any
reason**. The material probe firing on a `swap_image` row scores as a hit. The
floor therefore certifies "this signal fires on dirty rows", not "this signal
identifies the error it names" — which is what the precision-floor claim needs.

**Fix:** score each probe against the subtype it claims to detect
(`noise_subtype` is already in the frame) and report both figures. The joint
number stays meaningful for OR-fusion; the per-signal number is the one the
paper's claim rests on.

**Status:** DONE — `233ba2d`.

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

**Sweep finding — blocked on data.** The mechanism is confirmed (skin hue sits
inside both `orange` 14–40° and the `brown` rule's <50°), but the magnitude
cannot be measured here. `synthgen.render_product_image` (`tiger/data/synthgen.py:122`)
draws a flat-fill polygon on a 238–250 grey ground: no skin, no models. The
Fashion and ABO datasets are not in the repo (C5/E6). Changing the estimator with
no data that exercises the failure is editing blind.

**Status:** BLOCKED on Fashion/ABO data being available locally (C5)

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

**Sweep finding — blocked on data.** Confirmed in code (`lo, hi = 0.15, 0.85`),
but unfalsifiable on the only committed dataset: synthgen centres every shape
within ±4% of frame centre at 36–44% scale, so the central box is *correct* there.
A regression on the synthetic set would prove nothing either way.

**Status:** BLOCKED on Fashion/ABO data being available locally (C5)

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

**Sweep finding — blocked on data.** Confirmed in code, but synthgen emits square
images, so the distortion is identically zero on the committed dataset.

**Status:** BLOCKED on Fashion/ABO data being available locally (C5)

---

### B6 ⚑ · The two estimators never cross-check
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

**⚑ Architectural.** This is the system's only possible abstention on *value*
uncertainty. The γ-gate abstains on routing, Eq. 27–29 on schema and similarity;
nothing today abstains on "I do not know what colour this is", so a wrong-but-
in-domain value that raises CLIP similarity is committed silently (`E2`). Treat
this as a missing pipeline stage between Solver and Verify, not a threshold tweak.

B0's report already quantifies what this would buy before it is built.

**Status:** PARKED — ⚑ design change, decision required before implementation.
Size it from the B0 run; do not implement in a cleanup pass.

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

**Sweep finding — recoverable, and it should land first.** The suite was deleted
in `a7c1e72` ("Strip repository to absolute bare minimum"). Recovered contents:
12 files, **73** `def test_` functions (the README's "68" predates the last
additions). Restore with:

```bash
git checkout a7c1e72^ -- tests/ && .venv/bin/python -m pytest -q
```

This is purely additive — no runtime code changes — and it is the instrument that
proves subsequent fixes are architecture-preserving. **Do it before any other fix.**
Tests that fail on today's code are themselves findings and belong in this file.

**Fix:** restore the suite as above,
or remove the pytest config and drop the claim. Do not leave it declared-but-empty.
Also delete the surviving "✅ Unit tests written" line in
`paper_assets/tiger_project_doc.md` §11, which `f050639` missed (see E6).

**Status:** DONE — `62373dc`; 73 restored, all passing. Suite now at 101.

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
**Where:** `data/outputs/` is gitignored; `data/sample/` and `data/thresholds/` are absent

Every number in `paper_assets/` traces to Kaggle CSVs that survive in no
committed form. For a systems paper this is the reproducibility surface.

**Sweep correction:** `data/outputs/` is **not empty** — 41 files exist locally,
0 tracked. The artifacts that back the detection numbers are already on disk and
merely uncommitted, which makes this much cheaper than it reads. `data/sample/`
and `data/thresholds/` genuinely do not exist and must be regenerated (E6).

**Fix:** commit the summary CSVs (not the caches) under `paper_assets/results/`.

**Status:** TODO

---

### C6 · `requirements.txt` and `pyproject.toml` disagree
**Severity:** Low
**Where:** `requirements.txt` · `pyproject.toml:12-34`

`requirements.txt` pins `opencv-python`, `tqdm`, `requests` and `torchvision` —
none imported anywhere under `tiger/`. `matplotlib` is imported by
`tiger/viz.py:2` and declared in neither file, so a clean `pip install -e ".[dev]"`
cannot run `viz`.

**Fix:** delete `requirements.txt` in favour of the extras (or regenerate it from
them), and add a `viz` extra carrying `matplotlib`.

**Status:** TODO

---

### C7 · A 73 MB AWS installer is sitting in the project root
**Severity:** Low
**Where:** `awscliv2.zip`, `aws/` (untracked but present)

Alongside untracked `literature_review.md`, `related_work.tex`, `related_work.bib`
and `papers/`. Nothing distinguishes scratch from deliverable.

**Fix:** delete the installer and `aws/`; decide whether the literature files are
tracked deliverables and either commit them or add them to `.gitignore` explicitly.

**Status:** TODO

---

### C8 · Repair iterates a `set`, so the provenance log is not reproducible
**Severity:** Low
**Where:** `tiger/repair.py:95` — `for row_id in active_ids:`

Python string hashing is randomised per process. Outcomes are unaffected (the
pass reads `flagged` and `pool`, both fixed at pass start), but the provenance
log — the audit artefact the roadmap cites for rollback under 4.3 — comes out in
a different order on identical inputs.

**Fix:** `for row_id in sorted(active_ids):`

**Status:** DONE — `d1f22c3`.

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

### D3 · ~~Position bias in `check_t2v`~~ — WITHDRAWN
**Severity:** ~~Low~~ — does not apply
**Where:** `tiger/verify.py:176` · `tiger/vlm_judge.py:246`

**Original claim:** old/new images are presented in a fixed order, and
MLLM-as-a-Judge (ICML 2024) names position bias as a failure mode present even in
GPT-4V.

**Sweep finding: the premise does not hold for either implementation.**

1. `IndependentVerifier.check_t2v` is a **bi-encoder**. It embeds both images
   independently and compares `imgs[0] @ t` against `imgs[1] @ t`. Cosine
   similarity has no notion of presentation order — there is no position to bias.
2. `GeminiVLMJudge.check_t2v` sends **one image** (the proposed replacement).
   Position bias requires two items in an order; there is no ordering. This was
   already designed out — see `project_chronicle.md` Hiccup 3.

The citation is real and the failure mode is real for VLM judges in general; it
is simply not reachable in this code. Implementing "evaluate both orderings"
would add cost for a bias that cannot occur.

**Status:** WITHDRAWN — no action. Retained so the reasoning is not re-derived.

---

### D4 ⚑ · The two-pass loop never runs, so E3 has no behaviour of its own
**Severity:** High — a documented architectural edge that has never executed
**Where:** `tiger/repair.py:84` (mask) · `tiger/solver.py:224` (plan) · `tiger/repair.py:162` (apply)

```python
active_mask = flagged["flagged"].astype(bool) & ~flagged["row_id"].astype(str).isin(
    [rid for rid, oc in outcomes.items() if oc.final_status != "pending"])
```

An accepted repair sets `final_status = "repaired"` immediately, so pass 2
excludes it; escalated rows are excluded too. Pass 2 has nothing to act on and
`verify.max_passes: 2` is inert.

Downstream of that: `route()` returns E3 → direction `BOTH`; `plan_repair`
handles `("T2V","BOTH")` identically and returns a plan whose direction is
`"T2V"`; `repair.py` applies T2V and stops. **E3 is operationally
indistinguishable from E2** — the taxonomy has four classes and three behaviours.

**Consequence:** the "image first, then re-diagnose text" behaviour described in
`paper_concepts.md` §1, `tiger_project_doc.md` §9 and the E3 row of the taxonomy
table has never run. The re-route arrow in the README architecture diagram is
drawn but not wired. `mixed_swap_color` rows (2% of injected noise) get the image
swapped and keep the wrong colour.

**Fix:** pick a contract and implement it —
1. keep accepted rows `pending` so they re-enter the next pass, terminating when
   they are no longer flagged; or
2. give `BOTH` an explicit two-step plan inside one pass: T2V, re-embed, then V2T
   against the new image.

Either changes results. If adopted, it must land **before** the corrected
baseline run, not after.

**Status:** PARKED — ⚑ design change, decision required before implementation.
Do not fix in a cleanup pass.

---

### D5 · Five colours are in the domain *and* in the alias map
**Severity:** High — root cause of D6 and D7; land first
**Where:** `configs/schema.yaml:19-31`

`silver, gold, beige, navy, teal` appear in `color.values` **and** in
`color.aliases`, mapping to `gray, yellow, white, blue, green`. `Schema.in_domain`
normalises both sides so membership still works, but `Schema.domain("color")`
returns 17 raw entries of which 5 are semantic duplicates.

**Consequences:**
- Contrastive probes (`tiger/sieve.py:128`) build one caption per raw entry, so
  "silver" and "gray" compete as separate candidates for the same pixels. A grey
  product can lose its declared value to its own alias and fire a false
  `flag_probe_color`.
- Any comparison of a raw domain value against a normalised one breaks — D6, D7.

**Fix:** keep the ABO colours in `aliases` only, or in `values` only with the
alias removed. Then assert `set(values) ∩ set(aliases) == ∅` at schema load so it
cannot recur.

**Status:** TODO

---

### D6 · The independent verifier vetoes correct repairs on aliased colours
**Severity:** High — silently suppresses good repairs, in the column carrying the +13.3% claim
**Where:** `tiger/verify.py:174`

```python
pred = domain[int(np.argmax(np.stack(embs) @ img[0]))]   # raw domain value
return pred == self.schema.normalize(field, value)        # normalised value
```

`pred` is raw (`"navy"`); the right-hand side is normalised (`"blue"`). Whenever
the independent encoder's argmax lands on one of the five aliased colours,
`check_v2t` returns `False` and the repair is vetoed as a semantic failure.

**Consequence:** vetoes caused by this bug are indistinguishable in the results
from genuine wrong-direction catches — the exact quantity the Independent
Verifier ablation measures.

**Fix:** `return self.schema.normalize(field, pred) == self.schema.normalize(field, value)`.
**Requires D5** to remove the duplicate candidates as well.

**Status:** TODO

---

### D7 · `_title_color` manufactures false title-contradiction flags
**Severity:** Medium
**Where:** `tiger/sieve.py:174` returns raw · `tiger/sieve.py:165` compares against normalised

A "Navy Shirt" carrying `color: navy` yields `title_color="navy"`,
`declared="blue"`, and fires `flag_title_contradiction` — a signal the ablation
reports at precision 1.000.

**Fix:** normalise `_title_color`'s return value. **Requires D5.**

**Status:** TODO

---

### D8 · Category singularisation is fashion-only, so ABO captions are malformed
**Severity:** Medium — a competing explanation for the H11 underconfidence finding
**Where:** `tiger/text_views.py:23,35`

`CATEGORY_SINGULAR` covers `shirts/shoes/bags/hats` only. ABO categories fall
through to `rstrip("s")`, producing probe and LOO captions like *"a photo of a
red home_decor"* and *"a photo of a red electronic"* — an underscore token and a
non-noun, fed to CLIP as the whole prompt ensemble.

**Consequence:** every ABO probe margin, LOO delta, and Arbiter feature derived
from them was measured against a degraded prompt. H11 attributes ABO
underconfidence entirely to covariate shift; this is a mechanical alternative
that has not been ruled out.

**Fix:** add the ABO categories with real nouns (`home_decor → "home decoration"`,
`electronics → "electronic device"`), then re-run the confidence diagnostic
before drawing any conclusion from it.

**Status:** TODO

---

### D9 · `rstrip("s")` is the wrong primitive for singularisation
**Severity:** Low
**Where:** `tiger/text_views.py:35` · `tiger/generator.py:49`

`rstrip` strips *every* trailing `s`: `"dress" → "dre"`, `"glasses" → "glasse"`.
Harmless for the current four fashion categories, wrong for obvious next-vertical
candidates.

**Fix:** `removesuffix("s")`, or an explicit map with a fallback.

**Status:** TODO

---

### D10 · The SDXL prompt drops pattern and material before generation
**Severity:** Medium — a documented limitation is attributed to the wrong cause
**Where:** `tiger/generator.py:49-55`

```python
subject = f"{color} {cat_singular}" if color and cat_singular else caption
```

Only colour and category reach the prompt. `pattern` and `material` are accepted
as arguments and discarded.

**Consequence:** `honest_limitations.md` §2 and `paper_draft_materials.md` §4 both
explain the loss of "striped"/"printed" as diffusion models struggling with
fine-grained pattern adherence. The pattern never enters the prompt. The
limitation is real; the stated cause is a claim about SDXL that this code cannot
support.

**Fix:** include pattern and material in the prompt, regenerate the qualitative
grid, and re-assess whether the limitation survives. Correct §2 and §4 either way.

**Status:** TODO

---

### D11 · The Gemini prompt cache is keyed on full base64 images and is unbounded
**Severity:** Medium
**Where:** `tiger/vlm_judge.py:167` — `cache_key = json.dumps(parts, sort_keys=True)`

`parts` contains the base64-encoded image, so every key is the size of the image
and every image is retained for the process lifetime.
`project_chronicle.md` Hiccup 2 describes this as "an in-memory LRU cache"; it is
neither LRU nor bounded. On a 1,500-image run it holds the corpus twice.

**Fix:** key on `sha1(image bytes) + sha1(prompt)`; bound it (`functools.lru_cache`
or an explicit cap). Correct the chronicle's description of what was built.

**Status:** TODO

---

### D12 · `repair` always reports zero flagged products
**Severity:** Low
**Where:** `tiger/cli.py:589` — `total = summary.get("total", 0)`

`run_repair_cycle` emits `n_products`, `by_status` and `max_passes`. There is no
`total` key, so the user-facing summary always reads *"We attempted to repair the
0 flagged products."*

**Sweep note:** this is the *second* instance of the same bug. `8ba8d19` already
fixed the sibling at `tiger/eval/repair_ablation.py:135` (`total_attempted` now
sums `repaired + escalated`) as a drive-by. Only the `cli.py` site remains.

**Fix:** use `n_products`, or sum `by_status`.

**Status:** TODO

---

### D13 · Dead allocation in `encode_images`
**Severity:** Trivial
**Where:** `tiger/encoders.py:140-143`

A conditional `np.zeros` whose result is unconditionally overwritten thirty lines
later, with a comment already admitting it (`# simpler: resolve dim lazily below`).

**Fix:** delete it.

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

### E6 · The docs route readers to six paths that do not exist
**Severity:** Medium — first-contact credibility, and it is the reproducibility surface

| Referenced | By | Reality |
|---|---|---|
| `data/sample/` "committed" | `README.md`, `tiger_project_doc.md` §2 | absent, untracked |
| `data/thresholds/` "committed" | `README.md`, `tiger_project_doc.md` §2 | absent, untracked |
| `scripts/` legacy MVP "kept for reference" | `README.md`, `tiger_project_doc.md` §2 | deleted |
| `.github/workflows/ci.yml` | `ROADMAP_PROGRESS.md` 4.2 (marked ✅) | absent |
| `kaggle_workflow.ipynb` "in the repository root" | `README.md:132`, `tiger_project_doc.md` §7 Option B | absent; the real notebooks are `tiger.ipynb` and `tiger_abo.ipynb` |
| `[ROADMAP_PROGRESS.md](ROADMAP_PROGRESS.md)` | `README.md:17` | broken link; the file is in `paper_assets/` |

The two directories advertised as committed are precisely the two that would make
the repo reproducible. Pairs with C5.

**Fix:** commit the sample catalogue and locked thresholds (both small), correct
every path, drop the `scripts/` and `ci.yml` claims, and remove the surviving
"✅ Unit tests written" line in `tiger_project_doc.md` §11 (C3).

**Status:** TODO

---

### E7 · Three documents name three different final verifiers
**Severity:** Medium

`pipeline_architecture.md` §5 says "Gemini 3.7 Flash or Local SigLIP";
`project_chronicle.md` §2 and `ROADMAP_PROGRESS.md` decision 6 say SigLIP was
selected *over* Gemini; `README.md`'s end-to-end example is
`repair --seed 7 --vlm-judge`, i.e. Gemini. The README's own worked example does
not run the configuration the paper reports — and per A2 it currently cannot run
at all. "Gemini 3.7 Flash" is not a real model.

**Fix:** name SigLIP as the reported verifier everywhere, change the README
example to `--independent`, and describe Gemini as an evaluated alternative.

**Status:** TODO

---

### E8 · Ablation-table denominators move by 17 rows with no explanation
**Severity:** Medium — a reviewer will subtract these
**Where:** `paper_assets/paper_draft_materials.md` §1

| Config | Repaired | Escalated | Sum |
|---|---|---|---|
| No Arbiter | 168 | 266 | 434 |
| No Independent Verifier | 176 | 256 | 432 |
| No Generative Fallback | 148 | 269 | **417** |
| No Gamma Gate | 163 | 269 | 432 |
| Full System | 163 | 269 | 432 |

Same sample, five different totals. The cause is benign — `_evaluate_run`
(`tiger/eval/repair_ablation.py:96`) reports only `repaired` and `escalated`,
silently dropping `dismissed`, `acquire_image` and `unrepaired` — but the table
presents the two columns as exhaustive.

**Fix:** report all statuses, or add a `Total` column with a footnote. Regenerate
after A1.

**Status:** BLOCKED on A1

---

### E9 · The ablation credits LOO masking for the contrastive probes' result
**Severity:** Medium — the paper mis-credits its own strongest contribution
**Where:** `tiger/eval/ablation.py:91,140` · `paper_assets/paper_concepts.md` §2

`CONFIGS["probes_only"]` is `flag_probe_{color,material,pattern}` and
`CONFIGS["no_loo"]` is everything *except* those — both are the per-field
contrastive probes. They print as "LOO Probes Only" and "No LOO Masking", and the
takeaway line reads "Adding LOO Masking: +X F1".

Eq. 18 leave-one-out lives in `tiger/analyzer.py` and runs only on already-flagged
rows. **It contributes nothing to detection.** The mechanism that produces the
project's strongest verified result — mutate_text recall 0.267 → 0.853 — is the
contrastive probes. `paper_concepts.md` §2 presents LOO masking as the headline
detection contribution.

**Fix:** rename the ablation rows to "Probes Only" / "No Probes"; rewrite
`paper_concepts.md` §2 to credit the per-field contrastive probes for detection
and describe LOO as field attribution for routing. This correction runs in the
project's favour — the probe result is stronger and more novel than the LOO story.

**Status:** TODO

---

### E10 · `tiger_project_doc.md` §12 line counts are stale
**Severity:** Low

`cli.py` 505 → 732 · `vlm_judge.py` ~165 → 255 · `solver.py` 209 → 251 ·
`schema.py` 112 → 118. The table omits `generator.py`, `viz.py` and
`eval/repair_ablation.py` entirely.

**Fix:** regenerate, or drop the line-count column — it ages on every commit.

**Status:** TODO

---

### E11 · A planted always-flagged row sits inside the reported synthetic metrics
**Severity:** Medium — undisclosed evaluation artefact
**Where:** `tiger/data/synthgen.py:238` · `tiger/data/noise.py:274`

`forced_gen_000` is appended to the catalogue with sentinel `category: "uniforms"`
(not in `schema.categories`), `color: magenta` and `material: velvet` (neither in
Ω), hard-assigned to the **report** split, then unconditionally image-blanked by
the injector regardless of seed or configured rate.

It is guaranteed flagged (unknown category → `flag_text_out_of_domain`),
guaranteed to have no T2V candidate (its category is unique), and guaranteed to
fail Eq. 27 at verify. It exists to force the generative-fallback branch to run.

**Consequence:** defensible as a smoke test, but it is a hand-placed row inside
every reported detection number for the synthetic catalogue, and no document
mentions it.

**Fix:** move it to a fixture behind a smoke test, exclude it from reported
metrics, or disclose it in the evaluation setup — then confirm the synthetic
numbers are unchanged.

**Status:** TODO

---

### E12 · `swap_image` recall is quoted at two granularities as one number
**Severity:** Trivial

`README.md` quotes `swap_image 0.975` — the coarse **label**, and correct.
`tiger_project_doc.md` §8 places 0.975 in a table of **subtypes**, where the
verified value is 0.983 (59/60), with `swap_image_same_category` as the separate
0.950 row.

**Fix:** one number per granularity, each labelled with which it is.

**Status:** TODO

---

## Summary

| Section | Items | Done | Open | Parked / blocked / withdrawn |
|---|---|---|---|---|
| A. Measurement correctness | 9 | 8 | A2 (partial) | — |
| B. Repair accuracy | 8 | 1 | B3, B7 | B1, B2, B5 blocked on data · B4 blocked on B0 · **B6 parked ⚑** |
| C. Config & reproducibility | 8 | 2 | C1, C2, C4, C5, C6, C7 | — |
| D. Robustness & design | 13 | 0 | D1, D2, D5–D13 | **D4 parked ⚑** · D3 withdrawn |
| E. Documentation | 12 | 0 | E1, E2, E4–E12 | E3 blocked on A1 |
| **Total** | **50** | **11** | **31** | 2 parked · 4 blocked · 1 withdrawn |

**Section A is closed** apart from A2's model pin, which needs a live API key.
The measurement instrument is now trustworthy, so B and the remaining sections
can be measured against a baseline that means something.

**Test suite: 101 passing** (73 restored by C3, 28 added by the fixes above).
Every fix in this pass was verified against it; none changed behaviour the
suite did not already pin.

**Done in this pass:** `C3` → `A4` → `A1` → `A3`/`A3b` → `A5` → `A6` → `A8` →
`A7` → `A2` (partial) → `C8`. Section A is closed bar A2's model pin.

**Next action:** `D5 → D6/D7` (the alias collisions — they change pipeline
behaviour, so they belong before the baseline run) → re-run `ablate-repair` with
B0 instrumentation active → read the estimator attribution report → that report
decides whether `B7` (encoder path) is worth pursuing while B1/B2 remain blocked
on data.

Then the numbers move, and `E2`, `E3`, `E8` and the ⚑ decisions can be settled
against figures that mean something. Everything in `E` should wait for that run;
the current values in `paper_assets/` are the ones this pass invalidated.

`D4` and `B6` are **parked** — they are the two ⚑ design changes and are excluded
from the fix pass by decision, not by oversight.
