# TIGeR — Roadmap from here

Forward plan. The defect detail lives in `FIXES.md`; this is the ordering and
the critical path. Written 2026-09-10, after the measurement-harness pass.

**State:** 13 items done, 32 open, 2 parked, 1 withdrawn. 117 tests passing.
26 commits on `docs/fixes-backlog-audit`.

---

## The critical path, in one line

`Phase 1 (dataset) → Phase 2 (corrected run) → everything else`

Nothing in Phases 3–5 is worth starting before Phase 2 produces numbers. Nine
open items are literally or effectively blocked on it.

---

## Phase 0 — Measurement harness ✅ DONE

Section A closed apart from A2's model pin. The instrument that produces every
repair-side number was broken four separate ways; it now works, and the 73
restored tests plus 44 new ones pin it.

Consequence not yet acted on: **every number currently in `paper_assets/` was
produced by the broken harness.** They are not defensible until Phase 2 replaces
them.

---

## Phase 1 — Dataset migration ✅ DONE (`7541322`)

Verified end to end against the local ABO release: 11,839 products across 15
categories, balanced splits, 0 schema-invalid rows. 162 tests passing.

Two things the restored suite caught during this phase, both H10-class:
putting ABO types into `required_for_categories` on *raw* coverage (real
post-normalisation coverage is 45.5–92.8%, so requiring colour would flag every
clean-but-incomplete row as dirty), and dropping `shirts`, which would have
silently killed `attribute_drop` detection in the synthetic run.

Also removed a silent selection bias: the importer discarded any product whose
colour string did not resolve — 22% of the corpus — which meant `attribute_drop`
noise was the only way a row could ever lack a colour.

Decision taken: drop the Kaggle Myntra fashion set, use **official ABO**
(Collins et al., CVPR 2022) for both verticals.

| | Task |
|---|---|
| 1.1 | `CATEGORY_MAP` → two furnishing verticals. **A:** CHAIR, SOFA, TABLE, OTTOMAN, STOOL_SEATING, RUG, LAMP, LIGHT_FIXTURE, WALL_ART (~7.5k). **B:** FINERING, FINENECKLACEBRACELETANKLET, FINEEARRING, HANDBAG, SUITCASE, HAT (~5.9k). Explicit allowlist, not substring matching. |
| 1.2 | `schema.yaml`: category list, `required_for_categories`, and drop the two fashion-only size constraints (`shoes_have_numeric_sizes`, `apparel_letter_sizes`) — furniture has no size enum. |
| 1.3 | **Colour normalisation layer.** 273–1893 distinct free-text colour strings per type vs a 12-value Ω. Build against the observed distribution; `surface_forms` (D5) is the hook. Start on WALL_ART (21 distinct values) to validate. |
| 1.4 | Per-type cap. CELLULAR_PHONE_CASE is 64,853 items — 44% of the catalogue. Exclude, or subsample deliberately as an imbalance test for `reviewer_defense.md` Attack 9. |
| 1.5 | **D8** (category singularisation) moves onto the critical path: new categories need real nouns or every probe caption is malformed. |
| 1.6 | Delete `configs/config.yaml` — dead legacy MVP config, read by nothing, containing pre-fix values (F4's `copies_per_row: 15`, IQR thresholds, the flat noise model). It reads as live calibration and is a trap. |

**Why this vertical:** furniture and homeware give 56–88% material coverage on
real photos where wood, metal, glass and fabric look different. Fashion gave
`material_flip` recall of **0.200**, the weakest number in the paper. This makes
that signal measurable for the first time rather than a documented failure.

**Side effect:** with no fashion vertical, **B1 (skin counted as product colour)
stops being a defect that affects your evaluation.** Consider closing it as
out-of-scope rather than carrying it.

---

## Phase 2 — The corrected baseline run ⟵ everything waits here

Both notebooks are built and pushed.

| | Task |
|---|---|
| 2.1 | `tiger_corrected_run.ipynb` — synthetic catalogue → **detection** numbers. Retitled and narrowed: its Fashion half is gone, since both real-data verticals now come from ABO. The verified 0.267→0.853 probe result lives here. |
| 2.2 | `tiger_abo_corrected_run.ipynb` — the two ABO verticals → **repair** numbers. Phase 1 landed, so this is ready to run. |
| 2.3 | Record the manifest. γ, both seeds, the allowlist and both model IDs are written to `run_manifest.json` so C1 cannot recur. |

**Unblocks:** B4, E3, E8, and makes E1/E2/E4–E12 worth writing.

**Expect movement in an unpredictable direction.** Four independent defects fed
the old table and none biased it consistently.

---

## Phase 3 — Repair accuracy (Section B)

The actual goal. Sequence from the B0 estimator-attribution report, which
Phase 2 produces — it says whether the pixel path or the encoder path is the
bottleneck, so do not guess.

| | Task | Status |
|---|---|---|
| 3.1 | **B2** — product localisation. The fixed central 70% crop is exactly wrong for furniture, where a rug, a lamp and a sofa occupy different regions. | UNBLOCKED — `data/raw/abo/` is local |
| 3.2 | **B5** — aspect ratio destroyed before cropping. Only 36.2% of ABO images are square (21–2871 px). Compounds B2. | UNBLOCKED |
| 3.3 | **B4** — calibrate `pixel_color_confidence` against actual correctness instead of gating a raw pixel share at 0.55. | after B0 run |
| 3.4 | **B3** — the white-discount rule at 85%. | TODO |
| 3.5 | **B7** — CLIP is measured at 62% on attribute binding vs BLIP 88%. `compare_encoders` already supports the swap. | TODO |

B2, B5 and B3 are PIL/NumPy on images already on disk — cheap locally, no GPU.

---

## Phase 4 — Paper corrections (Section E)

Do **after** Phase 2 except where noted.

| | Task |
|---|---|
| 4.1 | **E2 — do this now, it does not need the run.** `reviewer_defense.md` Attack 3 calls the 47.4% "safely escalated to a human". Those rows were repaired and **committed with wrong values**. It presents a real error rate as a safety guarantee. Most exposed claim in the repo. |
| 4.2 | **E3 / E8** — §7.5 and H11 analyse an identity A1 manufactured. Withdraw or rewrite. If overlap survives the corrected run, prove it by intersecting escalated `row_id` sets; equal totals are not equal sets. |
| 4.3 | **E9** — the ablation credits LOO masking for the contrastive probes' result. LOO contributes nothing to detection. Correcting this runs in your favour: the probe result is stronger and more novel. |
| 4.4 | **E1, E6, E7, E10, E11, E12** — feature count, dead file paths, three different named verifiers, stale line counts, the undisclosed planted row, granularity mix. |
| 4.5 | **E4, E5** — reframe Attack 1 on reliability rather than cost, and answer the ARO objection (pairs with B7). |
| 4.6 | Citation: ABO as `collins2022abo` (added to `related_work.bib`). Resolve the licence discrepancy — the bucket ships CC BY 4.0, the AWS registry says CC BY-NC 4.0. NC would constrain Attack 1's commercial framing. |

---

## Phase 5 — Parked design decisions ⚑

Not bugs. Each needs a decision before it needs a patch.

| | |
|---|---|
| **B6** | Nothing abstains on *value* uncertainty. The γ-gate abstains on routing, Eq. 27–29 on schema and similarity; nothing abstains on "I do not know what colour this is", so a wrong-but-in-domain value that raises CLIP similarity is committed silently. Fixing it inserts a new stage between Solver and Verify. |
| **D4** | The two-pass loop never runs, so E3 has no behaviour distinct from E2 — four taxonomy classes, three implemented behaviours. The re-route arrow in the README has never executed. |

Both are worth doing. Neither should land in a cleanup pass, and B6 should be
sized from the B0 report rather than built blind.

---

## Housekeeping (any time)

`A2` pin a verified Gemini model ID (needs a live key) · `C1` γ consistency ·
`C2` `--gamma` flag · `C4` hardcoded generated-image path · `C5` commit the
evaluation artifacts · `C6` requirements/pyproject divergence · `C7` the 73 MB
AWS installer in the repo root · `D1` `class_weight="balanced"` vs the
calibration claim · `D2` document the verifier's asymmetric failure handling ·
`D9`–`D13` small correctness items.

---

## If you only do three things

1. **E2** — today, no run required. It is the claim most likely to be challenged.
2. **Phase 1 + Phase 2** — produce numbers that can be defended.
3. **B2 + B5** — cheap, local, no GPU, and they attack the component the whole
   pipeline is named after.
