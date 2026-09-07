"""End-to-end repair ablation (roadmap 5.1).

This runs the full repair cycle under different configurations to prove the
necessity of the repair-side components (Arbiter, VLM Judge, Generative Fallback, Gamma Gate).

To keep runtime manageable on Kaggle, this defaults to running on a small random
subset of the noisy dataset.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import random

import pandas as pd
import numpy as np

from tiger import arbiter as arbiter_mod
from tiger import repair as repair_mod
from tiger import verify as verify_mod
from tiger import sieve as sieve_mod
from tiger.encoders import ClipEncoder
from tiger.schema import Schema


class DummyArbiter(arbiter_mod.ArbiterModel):
    """Routes at random: the "no trained classifier" baseline.

    Two defects made the original neither random nor reproducible (A3, A3b):

      - it emitted an "E4" key. E4 is not a predicted class -- it is the state
        the gamma gate assigns (arbiter.CLASSES is E1/E2/E3/CLEAN). When E4 won
        the argmax, route() matched neither CLEAN nor E1, fell through to the
        final return, and was relabelled E3/BOTH. Those routes were not random,
        they were systematically pushed toward image swaps.
      - it hardcoded "CLEAN": 0.0, so the baseline could never dismiss a row.
        A router that cannot choose one of the four available outcomes is not a
        uniform baseline over the outcome space.
      - it drew from the global `random` module with no seed, so the row changed
        between runs and could not be reproduced for the paper.

    Now: a Dirichlet-flat draw over the four real classes from an instance-owned
    RNG, seeded from config so the seed travels with the run.
    """

    def __init__(self, *args, seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = random.Random(seed)
        self.training_meta = dict(self.training_meta or {})
        self.training_meta["dummy_arbiter_seed"] = seed

    def predict_proba(self, x: np.ndarray) -> dict[str, float]:
        draws = [self._rng.random() for _ in arbiter_mod.CLASSES]
        total = sum(draws) or 1.0
        return {c: d / total for c, d in zip(arbiter_mod.CLASSES, draws)}

    def to_json(self) -> str:
        # ArbiterModel.to_json serialises self.__dict__, which here holds a
        # random.Random. A baseline router is never a persisted artifact, so
        # fail loudly rather than emit something that cannot be loaded back.
        raise NotImplementedError(
            "DummyArbiter is an evaluation baseline and is not serialisable; "
            "persist the trained ArbiterModel instead."
        )


def run_repair_ablations(noisy_df: pd.DataFrame, enc: ClipEncoder, schema: Schema,
                         thr: sieve_mod.SieveThresholds, loo_stats: dict,
                         vcal: verify_mod.VerifyCalibration, trained_model: arbiter_mod.ArbiterModel,
                         cfg: dict, root: Path, generator=None, vlm_judge=None,
                         sample_size: int | None = 20) -> dict:
    
    # 1. Sample the dataset intelligently: grab ALL corrupted rows + some clean
    #    context rows so the repair pipeline has real material to work with.
    #    The old code sampled randomly and grabbed mostly clean rows.
    if "noise_label" in noisy_df.columns:
        noisy_rows = noisy_df[noisy_df["noise_label"] != "clean"]
        clean_rows = noisy_df[noisy_df["noise_label"] == "clean"]
        
        # Take up to sample_size corrupted rows
        n_noisy = len(noisy_rows) if sample_size is None else min(sample_size, len(noisy_rows))
        sampled_noisy = noisy_rows.sample(n=n_noisy, random_state=42)
        
        # Pad with clean rows so the Sieve has context (clean neighbours)
        # We need roughly 2x clean rows as noisy for realistic detection
        n_clean = min(n_noisy * 2, len(clean_rows))
        sampled_clean = clean_rows.sample(n=n_clean, random_state=42)
        
        noisy_sample = pd.concat([sampled_noisy, sampled_clean], ignore_index=True)
        print(f"  Sampled {n_noisy} corrupted + {n_clean} clean = {len(noisy_sample)} total rows")
    else:
        if sample_size is None:
            noisy_sample = noisy_df.copy()
        else:
            noisy_sample = noisy_df.sample(n=min(sample_size * 3, len(noisy_df)), random_state=42).reset_index(drop=True)
        print(f"  Sampled {len(noisy_sample)} rows (no noise_label column found)")
    
    # Build ground-truth lookup from the noise audit log
    seed = cfg.get("noise", {}).get("seed", 7)
    audit_path = root / cfg["data"]["processed_dir"] / f"noise_audit_seed{seed}.csv"
    if not audit_path.exists():
        # Try alternate naming convention
        audit_path = root / cfg["data"]["processed_dir"] / f"noise_audit_report_seed{seed}.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    truth_color = {}
    if not audit.empty:
        for _, r in audit[audit["field"] == "color"].iterrows():
            truth_color[str(r["row_id"])] = str(r["old_value"])
            
    def _evaluate_run(report: dict, final_df: pd.DataFrame,
                      config_name: str = "") -> tuple[dict, list[dict]]:
        """Aggregate metrics plus one diagnostic row per scored V2T repair.

        The per-case rows carry both estimator candidates and whether each would
        have been correct, so operator error can be attributed to the pixel path
        or the CLIP path independently of routing accuracy.
        """
        v2t_correct = v2t_total = 0
        cases: list[dict] = []
        after = final_df.set_index("row_id")
        repaired_c = report["summary"].get("by_status", {}).get("repaired", 0)
        escalated_c = report["summary"].get("by_status", {}).get("escalated", 0)

        for rid, oc in report["outcomes"].items():
            if oc["final_status"] != "repaired":
                continue
            if rid in truth_color:
                import json as _json
                attrs_str = after.at[rid, "attributes"]
                if isinstance(attrs_str, pd.Series):
                    attrs_str = attrs_str.iloc[0]
                new_color = str(_json.loads(attrs_str).get("color", ""))
                truth = truth_color[rid]
                ok = int(new_color == truth)
                v2t_total += 1
                v2t_correct += ok

                # the last V2T attempt carries the estimator attribution
                v2t_logs = [e for e in (oc.get("log") or []) if e.get("direction") == "V2T"]
                d = v2t_logs[-1] if v2t_logs else {}
                pv, bv = str(d.get("pixel_value") or ""), str(d.get("probe_value") or "")
                cases.append({
                    "config": config_name,
                    "row_id": rid,
                    "true_color": truth,
                    "written_color": new_color,
                    "correct": ok,
                    "value_source": d.get("value_source", ""),
                    "pixel_value": pv,
                    "pixel_conf": d.get("pixel_conf"),
                    "probe_value": bv,
                    "estimators_agree": d.get("estimators_agree"),
                    # counterfactuals: would each estimator alone have been right?
                    "pixel_correct": (int(pv == truth) if pv else None),
                    "probe_correct": (int(bv == truth) if bv else None),
                })

        return {
            "total_attempted": repaired_c + escalated_c,
            "repaired": repaired_c,
            "escalated": escalated_c,
            "color_accuracy": (v2t_correct / max(1, v2t_total)),
            "v2t_total": v2t_total
        }, cases

    results = {}
    v2t_cases: list[dict] = []
    sample_str = "the full dataset" if sample_size is None else f"a {sample_size}-item sample"
    print(f"\nRunning repair ablations on {sample_str}...")

    # Config 1: Full System
    print("1/5: Running 'Full System'...")
    cfg_full = copy.deepcopy(cfg)
    rep_full, rep_full_report = repair_mod.run_repair_cycle(
        noisy_sample, enc, schema, thr, loo_stats, vcal, trained_model, cfg_full, root,
        max_passes=2, independent=vlm_judge, generator=generator)
    results["full"], _c = _evaluate_run(rep_full_report, rep_full, "full")
    v2t_cases += _c

    # Config 2: No Arbiter (Random Routing)
    print("2/5: Running 'No Arbiter (Random Routing)'...")
    cfg_no_arbiter = copy.deepcopy(cfg)
    dummy_model = DummyArbiter(
        feature_names=trained_model.feature_names, classes=trained_model.classes,
        mean=trained_model.mean, scale=trained_model.scale, coef=trained_model.coef,
        intercept=trained_model.intercept,
        seed=int(cfg_no_arbiter.get("eval", {}).get("random_baseline_seed", 42)),
    )
    rep_no_arb, rep_no_arb_report = repair_mod.run_repair_cycle(
        noisy_sample, enc, schema, thr, loo_stats, vcal, dummy_model, cfg_no_arbiter, root,
        max_passes=2, independent=vlm_judge, generator=generator)
    results["no_arbiter"], _c = _evaluate_run(rep_no_arb_report, rep_no_arb, "no_arbiter")
    v2t_cases += _c

    # Config 3: No VLM Judge
    print("3/5: Running 'No VLM Judge'...")
    rep_no_vlm, rep_no_vlm_report = repair_mod.run_repair_cycle(
        noisy_sample, enc, schema, thr, loo_stats, vcal, trained_model, cfg_full, root,
        max_passes=2, independent=None, generator=generator)
    results["no_vlm"], _c = _evaluate_run(rep_no_vlm_report, rep_no_vlm, "no_vlm")
    v2t_cases += _c

    # Config 4: No Generative Fallback
    print("4/5: Running 'No Generative Fallback'...")
    rep_no_gen, rep_no_gen_report = repair_mod.run_repair_cycle(
        noisy_sample, enc, schema, thr, loo_stats, vcal, trained_model, cfg_full, root,
        max_passes=2, independent=vlm_judge, generator=None)
    results["no_gen"], _c = _evaluate_run(rep_no_gen_report, rep_no_gen, "no_gen")
    v2t_cases += _c

    # Config 5: No Gamma Gate (Set gamma threshold to 0.0 so everything passes)
    #
    # arbiter.route() reads cfg["arbiter"]["gamma"] (tiger/arbiter.py:197). This
    # previously wrote cfg["fusion"]["gamma"], a key nothing reads and which does
    # not exist in configs/tiger.yaml, so the ablation ran the identical
    # configuration as Full System -- which is why the two rows matched exactly.
    print("5/5: Running 'No Gamma Gate (Accept All Routes)'...")
    cfg_no_gamma = copy.deepcopy(cfg)
    cfg_no_gamma.setdefault("arbiter", {})["gamma"] = 0.0
    rep_no_gamma, rep_no_gamma_report = repair_mod.run_repair_cycle(
        noisy_sample, enc, schema, thr, loo_stats, vcal, trained_model, cfg_no_gamma, root,
        max_passes=2, independent=vlm_judge, generator=generator)
    results["no_gamma"], _c = _evaluate_run(rep_no_gamma_report, rep_no_gamma, "no_gamma")
    v2t_cases += _c

    results["_v2t_cases"] = v2t_cases
    return results


def format_repair_ablations(results: dict) -> str:
    lines = []
    lines.append("")
    lines.append("🛠️ Repair-Side Ablation Study: Component Impact")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"{'Configuration':<25s} | {'Repaired':<8s} | {'Escalated':<10s} | {'Restoration Acc (V2T)':<20s}")
    lines.append("-" * 75)

    friendly_names = {
        "full": "Full System",
        "no_arbiter": "No Arbiter (Random)",
        "no_vlm": "No VLM Judge",
        "no_gen": "No Generative Fallback",
        "no_gamma": "No Gamma Gate (γ=0)",
    }
    
    order = ["no_arbiter", "no_vlm", "no_gen", "no_gamma", "full"]

    for name in order:
        if name not in results:
            continue
        r = results[name]
        label = friendly_names.get(name, name)
        acc_str = f"{r['color_accuracy']:.1%} ({r['v2t_total']} cases)" if r['v2t_total'] > 0 else "N/A"
        lines.append(f"{label:<25s} | {r['repaired']:<8d} | {r['escalated']:<10d} | {acc_str:<20s}")

    lines.append("")
    lines.append("Key Takeaways:")
    if "full" in results and "no_arbiter" in results:
        delta_acc = results["full"]["color_accuracy"] - results["no_arbiter"]["color_accuracy"]
        lines.append(f"  • Trained Arbiter vs Random: {delta_acc:+.1%} restoration accuracy")
    if "full" in results and "no_gamma" in results:
        delta_esc = results["no_gamma"]["escalated"] - results["full"]["escalated"]
        lines.append(f"  • Gamma Gate: Safely escalated {delta_esc} uncertain cases instead of forcing bad repairs")
    if "full" in results and "no_vlm" in results:
        delta_acc = results["full"]["color_accuracy"] - results["no_vlm"]["color_accuracy"]
        lines.append(f"  • VLM Judge: Prevented coarse semantic swaps, improving accuracy by {delta_acc:+.1%}")
    if "full" in results and "no_gen" in results:
        delta_rep = results["full"]["repaired"] - results["no_gen"]["repaired"]
        lines.append(f"  • Generative Fallback: Successfully repaired {delta_rep} products that would otherwise be dead ends")

    return "\n".join(lines)


def save_v2t_diagnostics_csv(results: dict, out_path: str | Path) -> None:
    """Write one row per scored V2T repair with full estimator attribution."""
    cases = results.get("_v2t_cases") or []
    if cases:
        pd.DataFrame(cases).to_csv(out_path, index=False)


def format_v2t_diagnostics(results: dict, config: str = "full") -> str:
    """Decompose V2T restoration error into pixel-path vs CLIP-path error.

    Restoration accuracy alone cannot distinguish 'the Arbiter routed the wrong
    field' from 'the operator wrote the wrong value'. This reports, for the rows
    that were actually repaired, how often each estimator would have been right
    on its own -- which determines whether effort belongs in colours.py or in
    the encoder.
    """
    cases = [c for c in (results.get("_v2t_cases") or []) if c.get("config") == config]
    L = ["", f"🔬 V2T Estimator Attribution  (config: {config})", "=" * 75]
    if not cases:
        L.append("  no scored V2T cases -- nothing to attribute.")
        return "\n".join(L)

    n = len(cases)

    def pct(k, d):
        return f"{(k / d):.1%}" if d else "n/a"

    used = sum(c["correct"] for c in cases)
    L.append(f"  scored V2T repairs             : {n}")
    L.append(f"  accuracy as shipped            : {pct(used, n)}  ({used}/{n})")
    L.append("")

    # Which path supplied the written value, and how each did.
    L.append("  by path actually taken:")
    for src in ("pixel", "probe"):
        sub = [c for c in cases if c["value_source"] == src]
        if sub:
            k = sum(c["correct"] for c in sub)
            L.append(f"    {src:<6s} supplied {len(sub):>3d} values -> {pct(k, len(sub))} correct  ({k}/{len(sub)})")
    L.append("")

    # Counterfactual: how would each estimator have done on ALL of these rows?
    L.append("  counterfactual (same rows, one estimator throughout):")
    for key, label in (("pixel_correct", "always pixel"), ("probe_correct", "always probe")):
        sub = [c for c in cases if c.get(key) is not None]
        if sub:
            k = sum(c[key] for c in sub)
            L.append(f"    {label:<14s}: {pct(k, len(sub))}  ({k}/{len(sub)} scorable)")
    oracle = sum(int(bool(c.get("pixel_correct")) or bool(c.get("probe_correct"))) for c in cases)
    L.append(f"    {'either right':<14s}: {pct(oracle, n)}  ({oracle}/{n})  <- ceiling for any selection rule")
    L.append("")

    # Agreement: what an agree/disagree gate would buy (Step 2).
    agree = [c for c in cases if c.get("estimators_agree") is True]
    disagree = [c for c in cases if c.get("estimators_agree") is False]
    L.append("  agreement gate (prospective):")
    if agree:
        k = sum(c["correct"] for c in agree)
        L.append(f"    agree    : {len(agree):>3d} rows ({len(agree)/n:.0%} coverage) -> {pct(k, len(agree))} correct")
    if disagree:
        k = sum(c["correct"] for c in disagree)
        pk = sum(int(bool(c.get("pixel_correct"))) for c in disagree)
        bk = sum(int(bool(c.get("probe_correct"))) for c in disagree)
        L.append(f"    disagree : {len(disagree):>3d} rows ({len(disagree)/n:.0%} coverage) -> {pct(k, len(disagree))} correct")
        L.append(f"               when they disagree: pixel right {pct(pk, len(disagree))}, probe right {pct(bk, len(disagree))}")
    L.append("")
    L.append("  Reading: if 'agree' accuracy is high and 'disagree' is near chance, an")
    L.append("  agreement gate converts silent wrong-writes into escalations (trading")
    L.append("  coverage for precision). If 'always pixel' >> 'always probe' (or vice")
    L.append("  versa), fix the weaker estimator before touching routing.")
    return "\n".join(L)


def save_repair_ablations_csv(results: dict, out_path: str | Path) -> None:
    rows = []
    friendly_names = {
        "full": "Full System",
        "no_arbiter": "No Arbiter (Random)",
        "no_vlm": "No VLM Judge",
        "no_gen": "No Generative Fallback",
        "no_gamma": "No Gamma Gate (gamma=0)",
    }
    order = ["no_arbiter", "no_vlm", "no_gen", "no_gamma", "full"]
    for name in order:
        if name in results:
            r = results[name]
            rows.append({
                "Configuration": friendly_names.get(name, name),
                "Repaired": r["repaired"],
                "Escalated": r["escalated"],
                "Total Attempted": r["total_attempted"],
                "Color Accuracy": r["color_accuracy"],
                "V2T Cases": r["v2t_total"],
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
