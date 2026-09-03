"""End-to-end repair cycle: detect -> diagnose -> route -> plan -> apply -> verify.

Implements the closed loop the review found missing (A1.1): Verify's per-repair
verdict feeds back into routing, rejected repairs roll back and escalate, and
the cycle re-diagnoses so a row needing two fixes gets two (capped at two passes,
roadmap 2.5).

Per pass:
  1. compute signals on the working set (content-hash cache => only changed rows
     re-embed) and apply LOCKED thresholds;
  2. analyze still-flagged, not-yet-finalised rows -> evidence;
  3. Arbiter routes each; human/dismiss/acquire terminate the row;
  4. Solver plans a concrete repair; unplannable -> escalate;
  5. apply on a copy, re-embed the changed modality, Verify (Eq. 27-29);
  6. accept -> commit to the working set (re-checked next pass);
     reject -> roll back and escalate to the next tier (human).

Everything an accepted or rejected repair touched is written to a provenance
log (row, pass, action, plan, verdict) enabling rollback/audit (roadmap 4.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tiger import analyzer as analyzer_mod
from tiger import arbiter as arbiter_mod
from tiger import sieve as sieve_mod
from tiger import solver as solver_mod
from tiger import text_views
from tiger import verify as verify_mod
from tiger.encoders import ClipEncoder
from tiger.schema import Schema

TERMINAL_ACTIONS = {"human_review", "dismiss", "acquire_image", "generate_text"}


@dataclass
class RepairOutcome:
    row_id: str
    final_status: str            # repaired | escalated | dismissed | acquire_image | unrepaired
    passes_used: int = 0
    log: list = field(default_factory=list)  # per-attempt dicts

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _tau_for(thr: sieve_mod.SieveThresholds, category: str) -> float:
    return float(thr.sim.get(category, thr.sim_global)["tau"])


def run_repair_cycle(working: pd.DataFrame, encoder: ClipEncoder, schema: Schema,
                     thr: sieve_mod.SieveThresholds, loo_stats: dict,
                     vcal: verify_mod.VerifyCalibration, model: arbiter_mod.ArbiterModel,
                     cfg: dict, root: Path, max_passes: int = 2,
                     independent=None, generator=None) -> tuple[pd.DataFrame, dict]:
    working = working.reset_index(drop=True).copy()
    outcomes: dict[str, RepairOutcome] = {}

    def finalize(row_id: str, status: str, pass_i: int, entry: dict) -> None:
        oc = outcomes.setdefault(row_id, RepairOutcome(row_id, status))
        oc.final_status = status
        oc.passes_used = pass_i
        oc.log.append(entry)

    for pass_i in range(1, max_passes + 1):
        sig, arrays = sieve_mod.compute_signals(working, encoder, schema, cfg, root)
        flagged = sieve_mod.apply_thresholds(sig, thr)

        image_emb = arrays["image_emb"]
        caption_emb = arrays["caption_emb"]
        ok = np.asarray(arrays["image_ok"], dtype=bool)
        cat_ids = flagged["category"].astype(str).to_numpy()
        img_paths = flagged["image_path"].astype(str).tolist()
        product_ids = flagged["product_id"].astype(str).tolist()
        pool = solver_mod.CandidatePool(image_emb, product_ids, img_paths, ok)

        active_mask = flagged["flagged"].astype(bool) & ~flagged["row_id"].astype(str).isin(
            [rid for rid, oc in outcomes.items() if oc.final_status != "pending"])
        active_ids = set(flagged.loc[active_mask, "row_id"].astype(str))
        if not active_ids:
            break

        evidences = analyzer_mod.analyze(flagged, arrays, encoder, schema, thr, loo_stats, cfg, root)
        ev_by_id = {str(e.row_id): e.to_dict() for e in evidences}
        idx_by_id = {str(r): i for i, r in enumerate(flagged["row_id"].astype(str))}

        any_committed = False
        for row_id in active_ids:
            ev = ev_by_id.get(row_id)
            if ev is None:
                continue
            i = idx_by_id[row_id]
            category = str(flagged.at[i, "category"])
            route = arbiter_mod.route(ev, model, cfg)

            if route.action in TERMINAL_ACTIONS:
                status = {"dismiss": "dismissed", "acquire_image": "acquire_image"}.get(
                    route.action, "escalated")
                finalize(row_id, status, pass_i,
                         {"pass": pass_i, "action": route.action, "reason": route.reason,
                          "error_type": route.error_type})
                continue

            plan = solver_mod.plan_repair(ev, route, flagged.loc[i].to_dict(), pool,
                                          cat_ids, caption_emb[i], schema,
                                          generator=generator, root_path=root)
            if not plan.plannable:
                finalize(row_id, "escalated", pass_i,
                         {"pass": pass_i, "action": "human_review",
                          "reason": f"unplannable: {plan.notes}", "error_type": route.error_type})
                continue

            c_before = float(ev.get("sim_full")) if ev.get("sim_full") is not None else float("nan")
            tau = _tau_for(thr, category)
            eps = vcal.eps_for(category)

            if plan.direction == "V2T":
                attrs = text_views.parse_attrs(flagged.at[i, "attributes"])
                pr = solver_mod.apply_attr_patch(str(flagged.at[i, "title"]), category, attrs,
                                                 plan.patch, schema)
                if not pr.applied:
                    finalize(row_id, "escalated", pass_i,
                             {"pass": pass_i, "action": "human_review",
                              "reason": f"patch_refused: {pr.refusal_reason}",
                              "error_type": route.error_type})
                    continue
                new_caption = text_views.full_caption(category, pr.attrs)
                c_after = float(image_emb[i] @ encoder.encode_texts([new_caption])[0])
                indep = None
                if independent is not None:
                    fld = next(iter(plan.patch))
                    indep = independent.check_v2t(str((root / flagged.at[i, "image_path"]).resolve()),
                                                  category, fld, plan.patch[fld])
                verdict = verify_mod.verify_repair(row_id, category, pr.attrs, c_before, c_after,
                                                   tau, eps, schema, independent_ok=indep)
                entry = {"pass": pass_i, "direction": "V2T", "patch": plan.patch,
                         "value_source": plan.value_source,
                         "pixel_value": plan.pixel_value,
                         "pixel_conf": plan.pixel_conf,
                         "probe_value": plan.probe_value,
                         "estimators_agree": plan.estimators_agree,
                         "verdict": verdict.to_dict()}
                if verdict.accepted:
                    working.loc[working["row_id"].astype(str) == row_id,
                                ["title", "attributes", "canonical_text"]] = \
                        [pr.title, json.dumps(pr.attrs, ensure_ascii=False), pr.canonical_text]
                    outcomes.setdefault(row_id, RepairOutcome(row_id, "pending")).log.append(entry)
                    outcomes[row_id].final_status = "repaired"
                    outcomes[row_id].passes_used = pass_i
                    any_committed = True
                else:
                    finalize(row_id, "escalated", pass_i,
                             {**entry, "action": "human_review", "reason": verdict.reason})

            elif plan.direction == "T2V":
                cand_path = (root / plan.candidate_image_path).resolve()
                new_img_emb, cand_ok = encoder.encode_images([str(cand_path)])
                if not cand_ok[0]:
                    finalize(row_id, "escalated", pass_i,
                             {"pass": pass_i, "action": "human_review",
                              "reason": "candidate_image_unreadable", "error_type": route.error_type})
                    continue
                c_after = float(new_img_emb[0] @ caption_emb[i])
                attrs = text_views.parse_attrs(flagged.at[i, "attributes"])
                indep = None
                if independent is not None:
                    indep = independent.check_t2v(
                        str((root / flagged.at[i, "image_path"]).resolve()),
                        str(cand_path), text_views.full_caption(category, attrs))
                verdict = verify_mod.verify_repair(row_id, category, attrs, c_before, c_after,
                                                   tau, eps, schema, independent_ok=indep)
                entry = {"pass": pass_i, "direction": "T2V",
                         "candidate_product": plan.candidate_product_id,
                         "verdict": verdict.to_dict()}
                if verdict.accepted:
                    working.loc[working["row_id"].astype(str) == row_id, "image_path"] = \
                        plan.candidate_image_path
                    oc = outcomes.setdefault(row_id, RepairOutcome(row_id, "pending"))
                    oc.log.append(entry)
                    oc.final_status = "repaired"
                    oc.passes_used = pass_i
                    any_committed = True
                else:
                    finalize(row_id, "escalated", pass_i,
                             {**entry, "action": "human_review", "reason": verdict.reason})

        encoder.save_cache()
        if not any_committed:
            break  # nothing changed; a second pass would repeat the first

    # anything still flagged-and-unresolved after the cap -> human
    for row_id in list(outcomes):
        if outcomes[row_id].final_status == "pending":
            outcomes[row_id].final_status = "unrepaired"

    summary = {
        "n_products": int(working["product_id"].nunique()) if "product_id" in working else len(working),
        "by_status": pd.Series([o.final_status for o in outcomes.values()]).value_counts().to_dict(),
        "max_passes": max_passes,
    }
    return working, {"outcomes": {k: v.to_dict() for k, v in outcomes.items()}, "summary": summary}
