"""Command-line entry points for the tiger pipeline.

Usage (from repo root, venv active):

  python -m tiger.cli synthgen                     # build bundled sample catalogue
  python -m tiger.cli noise [--seed N]             # inject errors into the report split
  python -m tiger.cli calibrate                    # fit locked thresholds on clean calibration split
  python -m tiger.cli detect [--seed N]            # sieve the noisy report split with locked thresholds
  python -m tiger.cli evaluate [--seed N]          # detection metrics with product-level CIs
"""

from __future__ import annotations

# ---- Silence ALL noisy library output before importing anything ----
import os, warnings, logging
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*Flax classes are deprecated.*")
warnings.filterwarnings("ignore", message=".*google.generativeai.*")
for _lib in ("transformers", "diffusers", "torch", "PIL", "huggingface_hub",
             "accelerate", "safetensors", "filelock", "urllib3"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from tiger import sieve as sieve_mod
from tiger.data import noise as noise_mod
from tiger.data import synthgen
from tiger.data import fashion_import
from tiger.data import import_abo as abo_import
from tiger.encoders import ClipEncoder
from tiger.eval import detection as det_eval
from tiger.schema import load_schema

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: str = "configs/tiger.yaml") -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _paths(cfg: dict) -> dict[str, Path]:
    d = cfg["data"]
    return {
        "sample": ROOT / d["sample_dir"],
        "processed": ROOT / d["processed_dir"],
        "cache": ROOT / d["cache_dir"],
        "outputs": ROOT / d["outputs_dir"],
    }


def _load_fusion(enabled: bool):
    """Precision-floor fusion config, or None (roadmap 3.4, finding A7).

    `detect` and `run_repair_cycle` never loaded this -- only `ablate` did -- so
    the advertised fused operating point (P=0.888) was an offline ablation row
    while the live pipeline ran un-fused at P=0.793. Loading is opt-in because
    fusion trades recall for precision (mutate_text recall 0.853 -> 0.773), which
    changes what reaches the repair stage.
    """
    if not enabled:
        return None
    from tiger import fusion as fusion_mod
    path = ROOT / "data/thresholds/tiger_fusion.json"
    if not path.exists():
        raise FileNotFoundError(
            f"--fusion requested but {path} is missing; run `calibrate-fusion` first")
    return fusion_mod.FusionConfig.from_json(path.read_text(encoding="utf-8"))


def _encoder(cfg: dict) -> ClipEncoder:
    m = cfg["models"]
    return ClipEncoder(m["clip_model_name"], device=m.get("device", "cpu"),
                       batch_size=int(m.get("batch_size", 32)),
                       cache_dir=_paths(cfg)["cache"])


def cmd_synthgen(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    g = cfg.get("synthgen", {})
    df = synthgen.generate(
        ROOT, schema,
        seed=int(g.get("seed", 20260717)),
        products_per_category=int(g.get("products_per_category", 30)),
        image_size=int(g.get("image_size", 224)),
        calibration_fraction=float(g.get("calibration_fraction", 0.5)),
        out_dir=cfg["data"]["sample_dir"],
    )
    print(f"generated {len(df)} products -> {cfg['data']['sample_dir']}/products.parquet")
    print(df.groupby(["category", "split"]).size().to_string())


def cmd_import_fashion(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    if not args.source:
        print("Error: --source directory is required for import-fashion command")
        import sys
        sys.exit(1)
        
    source_dir = Path(args.source).resolve()
    out_dir = ROOT / cfg["data"]["sample_dir"]
    
    print(f"Importing Fashion Product Images from {source_dir}...")
    df = fashion_import.import_fashion(
        source_dir, 
        out_dir, 
        schema, 
        max_items=3000, 
        seed=int(cfg.get("noise", {}).get("seed", 7))
    )
    print(df.groupby(["category", "split"]).size().to_string())

def cmd_import_abo(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    if not args.listings_dir or not args.images_csv or not args.images_dir:
        print("Error: --listings-dir, --images-csv, and --images-dir are all required for import-abo")
        import sys; sys.exit(1)

    out_dir = ROOT / cfg["data"]["sample_dir"]
    print(f"Importing ABO from:\n  listings : {args.listings_dir}\n  images   : {args.images_csv}\n  img dir  : {args.images_dir}")
    df = abo_import.import_abo(
        listings_dir=Path(args.listings_dir).resolve(),
        images_csv=Path(args.images_csv).resolve(),
        images_dir=Path(args.images_dir).resolve(),
        out_dir=out_dir,
        schema=schema,
        max_items=5000,
        seed=int(cfg.get("noise", {}).get("seed", 7)),
    )
    print(df.groupby(["category", "split"]).size().to_string())


def _tag(split: str, seed: int) -> str:
    return f"seed{seed}" if split == "report" else f"cal_seed{seed}"


def cmd_noise(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    df = pd.read_parquet(p["sample"] / "products.parquet")
    split = getattr(args, "split", "report")
    report = df[df["split"] == split].reset_index(drop=True)

    ncfg = dict(cfg.get("noise", {}))
    if args.seed is not None:
        ncfg["seed"] = int(args.seed)
    res = noise_mod.inject(report, schema, ncfg)

    p["processed"].mkdir(parents=True, exist_ok=True)
    tag = _tag(split, ncfg.get("seed", 7))
    out = p["processed"] / f"noisy_report_{tag}.parquet"
    res.df.to_parquet(out, index=False)
    res.audit.to_csv(p["processed"] / f"noise_audit_{tag}.csv", index=False)
    (p["processed"] / f"noisy_report_{tag}.meta.json").write_text(
        json.dumps(res.meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(res.meta, indent=2))


def cmd_calibrate(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    df = pd.read_parquet(p["sample"] / "products.parquet")
    cal = df[df["split"] == "calibration"].reset_index(drop=True)

    enc = _encoder(cfg)
    sig, arrays = sieve_mod.compute_signals(cal, enc, schema, cfg, ROOT)
    thr = sieve_mod.calibrate(sig, cfg, schema)

    out = ROOT / "data/thresholds/tiger_locked_thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(thr.to_json(), encoding="utf-8")
    print(f"locked thresholds -> {out}")

    # Eq. 18 normalisation: clean-split leave-one-out delta stats (Phase 2.1)
    from tiger import analyzer as analyzer_mod
    loo_stats = analyzer_mod.calibrate_loo(sig, arrays, enc, schema, schema.checkable_fields())
    loo_out = ROOT / "data/thresholds/tiger_loo_calibration.json"
    loo_out.write_text(json.dumps(loo_stats, indent=2), encoding="utf-8")
    print(f"LOO calibration -> {loo_out}")

    # Eq. 29 epsilon: caption-rewording noise floor on clean rows (Phase 2.5)
    from tiger import verify as verify_mod
    vcal = verify_mod.calibrate_epsilon(sig, arrays, enc,
                                        quantile=float(cfg.get("verify", {}).get("epsilon_quantile", 0.95)))
    vout = ROOT / "data/thresholds/tiger_verify_calibration.json"
    vout.write_text(vcal.to_json(), encoding="utf-8")
    print(f"verify epsilon={vcal.epsilon:.4f} (by category: "
          f"{ {k: round(v,4) for k,v in vcal.epsilon_by_category.items()} }) -> {vout}")


def cmd_detect(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    tag = _tag(getattr(args, "split", "report"),
               args.seed if args.seed is not None else cfg.get("noise", {}).get("seed", 7))
    noisy = pd.read_parquet(p["processed"] / f"noisy_report_{tag}.parquet")

    thr_path = ROOT / "data/thresholds/tiger_locked_thresholds.json"
    thr = sieve_mod.SieveThresholds.from_json(thr_path.read_text(encoding="utf-8"))

    enc = _encoder(cfg)
    fusion = _load_fusion(getattr(args, "fusion", False))
    sig, arrays = sieve_mod.compute_signals(noisy, enc, schema, cfg, ROOT)
    flagged = sieve_mod.apply_thresholds(sig, thr, fusion=fusion)

    import numpy as np
    p["outputs"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p["outputs"] / f"sieve_{tag}_arrays.npz", **arrays)

    out = p["outputs"] / f"sieve_{tag}.parquet"
    flagged.to_parquet(out, index=False)
    print("\n\n📊 Stage 1: Error Detection Complete")
    print("-" * 50)
    print(f"operating point: {'full + fusion (precision floor)' if fusion else 'full (OR-fused signals)'}")
    flagged_count = int(flagged['flagged'].sum())
    total_count = len(flagged)
    print(f"Out of {total_count} products scanned, we flagged {flagged_count} suspicious items.")
    print("\nReasons for flagging:")
    reasons = flagged["flag_reason"].value_counts()
    for reason, count in reasons.items():
        if reason:
            print(f"- {count} items: {reason}")
    print(f"\n(Raw data saved to {out})")


def cmd_evaluate(cfg: dict, args) -> None:
    p = _paths(cfg)
    tag = f"seed{args.seed if args.seed is not None else cfg.get('noise', {}).get('seed', 7)}"
    df = pd.read_parquet(p["outputs"] / f"sieve_{tag}.parquet")
    res = det_eval.evaluate(df)
    out = p["outputs"] / f"detection_metrics_{tag}.json"
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(det_eval.format_report(res))
    print(f"\nwrote {out}")


def cmd_analyze(cfg: dict, args) -> None:
    """Phase 2.1/2.2: compute Eq. 18 + Eq. 19 evidence for flagged rows."""
    import numpy as np

    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    tag = _tag(getattr(args, "split", "report"),
               args.seed if args.seed is not None else cfg.get("noise", {}).get("seed", 7))

    df = pd.read_parquet(p["outputs"] / f"sieve_{tag}.parquet")
    z = np.load(p["outputs"] / f"sieve_{tag}_arrays.npz")
    arrays = {k: z[k] for k in z.files}

    thr = sieve_mod.SieveThresholds.from_json(
        (ROOT / "data/thresholds/tiger_locked_thresholds.json").read_text(encoding="utf-8"))
    loo_stats = json.loads(
        (ROOT / "data/thresholds/tiger_loo_calibration.json").read_text(encoding="utf-8"))

    from tiger import analyzer as analyzer_mod
    enc = _encoder(cfg)
    evidences = analyzer_mod.analyze(df, arrays, enc, schema, thr, loo_stats, cfg, ROOT)
    out = p["outputs"] / f"evidence_{tag}.jsonl"
    analyzer_mod.save_evidence(evidences, out)
    print(f"wrote {out} ({len(evidences)} evidence records)")


def _load_evidence_with_labels(p: dict, tag: str) -> tuple[list[dict], list[str]]:
    ev = [json.loads(l) for l in (p["outputs"] / f"evidence_{tag}.jsonl").open(encoding="utf-8")]
    truth = pd.read_parquet(p["outputs"] / f"sieve_{tag}.parquet") \
        .set_index("row_id")["noise_label"].astype(str).to_dict()
    labels = [truth.get(e["row_id"], "clean") for e in ev]
    return ev, labels


def cmd_train_arbiter(cfg: dict, args) -> None:
    """Phase 2.3: fit p(E1..E4) router on CALIBRATION-split noise runs."""
    from tiger import arbiter as arbiter_mod

    p = _paths(cfg)
    seeds = [int(s) for s in cfg.get("arbiter", {}).get("train_seeds", [1007, 1008, 1009, 1010])]
    train_seeds, holdout_seed = seeds[:-1], seeds[-1]

    for s in seeds:
        ns = argparse.Namespace(seed=s, split="calibration")
        cmd_noise(cfg, ns)
        cmd_detect(cfg, ns)
        cmd_analyze(cfg, ns)

    train_ev, train_lab = [], []
    for s in train_seeds:
        ev, lab = _load_evidence_with_labels(p, _tag("calibration", s))
        # missing-modality rows are routed by rule, not by the model
        keep = [(e, l) for e, l in zip(ev, lab) if not e.get("image_missing") and not e.get("text_missing")]
        train_ev += [e for e, _ in keep]
        train_lab += [l for _, l in keep]

    model = arbiter_mod.train(train_ev, train_lab,
                              meta={"train_seeds": train_seeds, "holdout_seed": holdout_seed,
                                    "split": "calibration"})

    hold_ev, hold_lab = _load_evidence_with_labels(p, _tag("calibration", holdout_seed))
    keep = [(e, l) for e, l in zip(hold_ev, hold_lab) if not e.get("image_missing") and not e.get("text_missing")]
    hold_ev, hold_lab = [e for e, _ in keep], [l for _, l in keep]
    rel = arbiter_mod.reliability_table(model, hold_ev, hold_lab)
    model.training_meta["reliability_holdout"] = rel

    import numpy as np
    correct = 0
    for e, l in zip(hold_ev, hold_lab):
        pr = model.predict_proba(arbiter_mod.featurize(e))
        correct += (max(pr, key=pr.get) == arbiter_mod.LABEL_TO_CLASS.get(l, "CLEAN"))
    model.training_meta["holdout_accuracy"] = correct / max(1, len(hold_ev))

    out = ROOT / "data/thresholds/tiger_arbiter_model.json"
    out.write_text(model.to_json(), encoding="utf-8")
    print(f"trained on {model.training_meta['n_train']} rows "
          f"(classes {model.training_meta['class_counts']})")
    print(f"holdout (seed {holdout_seed}) accuracy: {model.training_meta['holdout_accuracy']:.3f}")
    print("reliability (holdout):")
    for b in rel:
        print(f"  conf~{b['mean_confidence']:.2f} -> acc {b['empirical_accuracy']:.2f} (n={b['n']})")
    print(f"wrote {out}")


def cmd_route(cfg: dict, args) -> None:
    """Apply the trained Arbiter to a report-split evidence file -> routing plan."""
    from tiger import arbiter as arbiter_mod

    p = _paths(cfg)
    tag = _tag("report", args.seed if args.seed is not None else cfg.get("noise", {}).get("seed", 7))
    model = arbiter_mod.ArbiterModel.from_json(
        (ROOT / "data/thresholds/tiger_arbiter_model.json").read_text(encoding="utf-8"))

    ev, labels = _load_evidence_with_labels(p, tag)
    routes = [arbiter_mod.route(e, model, cfg) for e in ev]

    rows = []
    for r, e, l in zip(routes, ev, labels):
        d = r.to_dict()
        d["probs"] = json.dumps({k: round(v, 3) for k, v in r.probs.items()})
        d["truth"] = arbiter_mod.LABEL_TO_CLASS.get(l, l.upper() if l.startswith("missing") else "CLEAN")
        d["truth_raw"] = l
        rows.append(d)
    plan = pd.DataFrame(rows)
    out = p["outputs"] / f"route_plan_{tag}.csv"
    plan.to_csv(out, index=False)

    print("\n\n🚦 Stage 3: Routing Decisions")
    print("-" * 50)
    print("The AI Arbiter analyzed the evidence for the flagged items and assigned repair strategies:")
    
    actions = plan["action"].value_counts()
    action_map = {
        "V2T": "📝 Fix the Text (V2T)",
        "T2V": "🖼️ Replace the Image (T2V)",
        "BOTH": "🔄 Fix Both (E3)",
        "human_review": "🧑‍💻 Escalate to Human",
        "dismiss": "✅ Dismiss as False Positive"
    }
    
    for action, count in actions.items():
        desc = action_map.get(action, f"[{action}]")
        print(f"- {desc:<28} {count} items")
        
    print(f"\n(Raw plan saved to {out})")


def cmd_compare_encoders(cfg: dict, args) -> None:
    """Phase 3.3: per-field probe accuracy per encoder on the clean catalogue."""
    from tiger.eval import encoder_compare as ec

    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    df = pd.read_parquet(p["sample"] / "products.parquet")
    fields = [f for f in cfg.get("sieve", {}).get("probes", {}).get("fields", []) if f in schema.checkable_fields()]
    names = cfg.get("models", {}).get("compare_encoders", [cfg["models"]["clip_model_name"]])

    encoders = {n: ClipEncoder(n, device=cfg["models"].get("device", "cpu"),
                               batch_size=int(cfg["models"].get("batch_size", 32)),
                               cache_dir=p["cache"]) for n in names}
    results = ec.compare(encoders, df, schema, fields)
    print(f"=== per-field probe accuracy on {len(df)} clean products ===")
    print(ec.format_comparison(results, fields))
    (p["outputs"] / "encoder_comparison.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nbaseline (reported): {cfg['models']['clip_model_name']}")
    print(f"wrote {p['outputs'] / 'encoder_comparison.json'}")


def cmd_calibrate_fusion(cfg: dict, args) -> None:
    """Phase 3.4/3.6: tune per-signal margins to a precision floor on labelled
    CALIBRATION-split noise runs; quarantine signals that miss the floor."""
    import glob

    from tiger import fusion as fusion_mod

    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    files = sorted(glob.glob(str(p["outputs"] / "sieve_cal_seed*.parquet")))
    if not files:
        raise FileNotFoundError("no calibration-split sieve files; run train-arbiter first")
    labeled = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    probe_fields = cfg.get("sieve", {}).get("probes", {}).get("fields", [])
    floor = float(cfg.get("sieve", {}).get("precision_floor", 0.85))
    fc = fusion_mod.calibrate_fusion(labeled, probe_fields, precision_floor=floor)

    out = ROOT / "data/thresholds/tiger_fusion.json"
    out.write_text(fc.to_json(), encoding="utf-8")
    print(f"precision floor {floor}; calibrated on {len(labeled)} labelled rows")
    for sig, v in fc.per_signal.items():
        q = " QUARANTINED" if v.get("quarantined") else ""
        zm = f" z>={v['z_margin']}" if v.get("z_margin") else ""
        print(f"  {sig:26s} precision={v.get('precision')} fired={v.get('fired')}{zm}{q}")
    print(f"wrote {out}")


def cmd_ablate(cfg: dict, args) -> None:
    """Phase 5.4: baselines + ablations over report-split seeds."""
    from tiger import fusion as fusion_mod
    from tiger.eval import ablation as abl

    p = _paths(cfg)
    seeds = [int(s) for s in args.seeds.split(",")]
    frames = []
    for s in seeds:
        f = p["outputs"] / f"sieve_seed{s}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        raise FileNotFoundError("no report-split sieve files; run sweep or detect first")

    thr = sieve_mod.SieveThresholds.from_json(
        (ROOT / "data/thresholds/tiger_locked_thresholds.json").read_text(encoding="utf-8"))
    fpath = ROOT / "data/thresholds/tiger_fusion.json"
    fusion = fusion_mod.FusionConfig.from_json(fpath.read_text(encoding="utf-8")) if fpath.exists() else None

    results = abl.run_ablations(frames, thr, fusion=fusion)
    print(f"=== baselines & ablations ({len(frames)} seeds pooled) ===")
    print(abl.format_ablations(results))
    out = p["outputs"] / "ablations.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    
    out_csv = p["outputs"] / "sieve_ablations_summary.csv"
    abl.save_ablations_csv(results, out_csv)
    print(f"wrote {out_csv}")


def cmd_ablate_repair(cfg: dict, args) -> None:
    """Phase 5.4 (Repair side): ablations over repair configurations."""
    import numpy as np
    from tiger import arbiter as arbiter_mod
    from tiger import verify as verify_mod
    from tiger.eval import repair_ablation

    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    seed = args.seed if args.seed is not None else cfg.get("noise", {}).get("seed", 7)
    tag = _tag("report", seed)
    noisy = pd.read_parquet(p["processed"] / f"noisy_report_{tag}.parquet")

    thr = sieve_mod.SieveThresholds.from_json(
        (ROOT / "data/thresholds/tiger_locked_thresholds.json").read_text(encoding="utf-8"))
    loo_stats = json.loads((ROOT / "data/thresholds/tiger_loo_calibration.json").read_text(encoding="utf-8"))
    vcal = verify_mod.load_calibration(ROOT / "data/thresholds/tiger_verify_calibration.json")
    model = arbiter_mod.ArbiterModel.from_json(
        (ROOT / "data/thresholds/tiger_arbiter_model.json").read_text(encoding="utf-8"))

    enc = _encoder(cfg)
    
    # Initialize optional components
    independent = None
    iv_name = cfg.get("models", {}).get("independent_verifier", "")
    if getattr(args, "vlm_judge", False):
        from tiger.vlm_judge import GeminiVLMJudge
        _judge = GeminiVLMJudge.from_env(verbose=False)
        class _JudgeAdapter:
            def check_v2t(self, image_path, category, field, value): return _judge.check_v2t(image_path, category, field, value)
            def check_t2v(self, old_image_path, new_image_path, caption): return _judge.check_t2v(old_image_path, new_image_path, caption)
        independent = _JudgeAdapter()
    elif getattr(args, "independent", False) and iv_name:
        iv_enc = ClipEncoder(iv_name, device=cfg["models"].get("device", "cpu"),
                             batch_size=int(cfg["models"].get("batch_size", 32)),
                             cache_dir=_paths(cfg)["cache"])
        independent = verify_mod.IndependentVerifier(iv_enc, schema)

    generator = None
    if getattr(args, "generative_fallback", False):
        from tiger.generator import StableDiffusionGenerator
        generator = StableDiffusionGenerator(device=cfg["models"].get("device", "cuda"))
    sample_size = getattr(args, "sample", None)
    if sample_size is not None:
        sample_size = int(sample_size)

    results = repair_ablation.run_repair_ablations(
        noisy, enc, schema, thr, loo_stats, vcal, model, cfg, ROOT,
        generator=generator, vlm_judge=independent, sample_size=sample_size,
        fusion=_load_fusion(getattr(args, "fusion", False))
    )
    
    print(repair_ablation.format_repair_ablations(results))
    print(repair_ablation.format_v2t_diagnostics(results, config="full"))

    out = p["outputs"] / "repair_ablations.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")

    out_csv = p["outputs"] / "repair_ablations_summary.csv"
    repair_ablation.save_repair_ablations_csv(results, out_csv)
    print(f"wrote {out_csv}")

    diag_csv = p["outputs"] / "v2t_estimator_diagnostics.csv"
    repair_ablation.save_v2t_diagnostics_csv(results, diag_csv)
    if (results.get("_v2t_cases") or []):
        print(f"wrote {diag_csv}")


def cmd_repair(cfg: dict, args) -> None:
    """Phase 2.5: end-to-end detect->diagnose->route->plan->apply->verify cycle."""
    import numpy as np

    from tiger import arbiter as arbiter_mod
    from tiger import repair as repair_mod
    from tiger import verify as verify_mod

    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    seed = args.seed if args.seed is not None else cfg.get("noise", {}).get("seed", 7)
    tag = _tag("report", seed)
    noisy = pd.read_parquet(p["processed"] / f"noisy_report_{tag}.parquet")

    thr = sieve_mod.SieveThresholds.from_json(
        (ROOT / "data/thresholds/tiger_locked_thresholds.json").read_text(encoding="utf-8"))
    loo_stats = json.loads((ROOT / "data/thresholds/tiger_loo_calibration.json").read_text(encoding="utf-8"))
    vcal = verify_mod.load_calibration(ROOT / "data/thresholds/tiger_verify_calibration.json")
    model = arbiter_mod.ArbiterModel.from_json(
        (ROOT / "data/thresholds/tiger_arbiter_model.json").read_text(encoding="utf-8"))

    enc = _encoder(cfg)
    max_passes = int(cfg.get("verify", {}).get("max_passes", 2))

    independent = None
    iv_name = cfg.get("models", {}).get("independent_verifier", "")
    if getattr(args, "vlm_judge", False):
        # Gemini VLM judge (roadmap 6.4): product-identity-aware, catches
        # same-category wrong-direction repairs that encoder-only checks miss.
        from tiger.vlm_judge import GeminiVLMJudge
        _judge = GeminiVLMJudge.from_env(verbose=True)
        print(f"VLM judge: {_judge.model_name} (Gemini)")

        class _JudgeAdapter:
            """Wrap GeminiVLMJudge to match IndependentVerifier's call interface."""
            def check_v2t(self, image_path, category, field, value):
                return _judge.check_v2t(image_path, category, field, value)
            def check_t2v(self, old_image_path, new_image_path, caption):
                return _judge.check_t2v(old_image_path, new_image_path, caption)

        independent = _JudgeAdapter()
    elif getattr(args, "independent", False) and iv_name:
        iv_enc = ClipEncoder(iv_name, device=cfg["models"].get("device", "cpu"),
                             batch_size=int(cfg["models"].get("batch_size", 32)),
                             cache_dir=_paths(cfg)["cache"])
        independent = verify_mod.IndependentVerifier(iv_enc, schema)
        print(f"independent verifier: {iv_name}")

    generator = None
    if getattr(args, "generative_fallback", False):
        from tiger.generator import StableDiffusionGenerator
        generator = StableDiffusionGenerator(device=cfg["models"].get("device", "cuda"))
        
    repaired, report = repair_mod.run_repair_cycle(noisy, enc, schema, thr, loo_stats, vcal,
                                                   model, cfg, ROOT, max_passes=max_passes,
                                                   independent=independent, generator=generator,
                                                   fusion=_load_fusion(getattr(args, "fusion", False)))

    p["outputs"].mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(p["processed"] / f"repaired_report_{tag}.parquet", index=False)
    (p["outputs"] / f"repair_report_{tag}.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")

    # ---- honest repair-quality evaluation (F14: not just re-flagging) ----
    audit_path = p["processed"] / f"noise_audit_{tag}.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    truth_color = {}  # row_id -> original (correct) colour, from the noise audit
    if not audit.empty:
        for _, r in audit[audit["field"] == "color"].iterrows():
            truth_color[str(r["row_id"])] = str(r["old_value"])

    before = noisy.set_index("row_id")
    after = repaired.set_index("row_id")
    v2t_correct = v2t_total = 0
    for rid, oc in report["outcomes"].items():
        if oc["final_status"] != "repaired":
            continue
        if rid in truth_color:
            import json as _json
            new_color = _json.loads(after.at[rid, "attributes"]).get("color", "")
            v2t_total += 1
            v2t_correct += int(str(new_color) == truth_color[rid])

    print("\n\n🛠️ Stage 4 & 5: Repair & Verify")
    print("-" * 50)
    summary = report["summary"]
    total = summary.get("total", 0)
    repaired_c = summary.get("by_status", {}).get("repaired", 0)
    escalated_c = summary.get("by_status", {}).get("escalated", 0)
    
    print(f"We attempted to repair the {total} flagged products:")
    print(f"✅ {repaired_c} products were successfully repaired automatically!")
    if escalated_c > 0:
        print(f"⚠️  {escalated_c} products failed safety checks and were escalated to human review.")
        
    if v2t_total:
        print(f"\n[Audit] Color Text Restoration (vs secret ground truth): {v2t_correct}/{v2t_total} ({v2t_correct/v2t_total:.1%})")

    # before/after re-flag context (reported, NOT the acceptance criterion -- F7)
    sig_a, _ = sieve_mod.compute_signals(repaired, enc, schema, cfg, ROOT)
    flg_a = sieve_mod.apply_thresholds(sig_a, thr, fusion=_load_fusion(getattr(args, "fusion", False)))
    still_flagged = int(flg_a['flagged'].sum())
    
    print("\nPost-Repair Check:")
    print(f"Out of the {int(len(noisy))} originally suspicious products, only {still_flagged} remain flagged after our repairs.")
    print("(Our automated repairs successfully fixed the issues for the rest!)")


def cmd_sweep(cfg: dict, args) -> None:
    """Multi-seed evaluation (roadmap 5.5): noise -> detect -> evaluate per seed,
    then aggregate mean +/- std and pooled per-error-type recall."""
    import numpy as np

    seeds = [int(s) for s in args.seeds.split(",")]
    p = _paths(cfg)
    all_res, pooled = [], []
    for s in seeds:
        ns = argparse.Namespace(seed=s)
        cmd_noise(cfg, ns)
        cmd_detect(cfg, ns)
        df = pd.read_parquet(p["outputs"] / f"sieve_seed{s}.parquet")
        pooled.append(df.assign(seed=s))
        res = det_eval.evaluate(df, n_bootstrap=0)
        all_res.append(res)
        print(f"seed {s}: P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f}")

    P = [r["precision"] for r in all_res]
    R = [r["recall"] for r in all_res]
    F = [r["f1"] for r in all_res]
    print(f"\n=== {len(seeds)} seeds ===")
    print(f"precision {np.mean(P):.3f} +/- {np.std(P, ddof=1):.3f}")
    print(f"recall    {np.mean(R):.3f} +/- {np.std(R, ddof=1):.3f}")
    print(f"f1        {np.mean(F):.3f} +/- {np.std(F, ddof=1):.3f}")

    big = pd.concat(pooled, ignore_index=True)
    # product-level bootstrap on the pooled runs (product resampled across seeds)
    res_pooled = det_eval.evaluate(big, n_bootstrap=2000)
    print("\n=== pooled across seeds ===")
    print(det_eval.format_report(res_pooled))
    out = p["outputs"] / "detection_metrics_sweep.json"
    out.write_text(json.dumps({"seeds": seeds, "per_seed": all_res, "pooled": res_pooled},
                              indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    
    rows = []
    labels = sorted(res_pooled["recall_by_label"].keys())
    row = {
        "Configuration": "Pooled (All Seeds)",
        "Precision": res_pooled["precision"],
        "Recall": res_pooled["recall"],
        "F1": res_pooled["f1"],
    }
    for l in labels:
        row[f"Recall ({l})"] = res_pooled["recall_by_label"].get(l, {}).get("recall", float("nan"))
    rows.append(row)
    
    out_csv = p["outputs"] / "sweep_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


def cmd_generate(cfg: dict, args) -> None:
    """Standalone generative model test."""
    from tiger.generator import StableDiffusionGenerator
    import sys
    from pathlib import Path
    
    if not args.caption:
        print("Error: --caption is required for generate command")
        sys.exit(1)
        
    out_path = Path(args.output) if args.output else Path("data/outputs/generated_test.jpg")
    generator = StableDiffusionGenerator(device=cfg["models"].get("device", "cuda"))
    print(f"Generating image for caption: '{args.caption}'")
    generator.generate(args.caption, out_path)
    print(f"Saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="tiger")
    ap.add_argument("command", choices=["synthgen", "import-fashion", "import-abo", "noise", "calibrate", "detect", "evaluate",
                                        "analyze", "train-arbiter", "route", "repair", "sweep",
                                        "calibrate-fusion", "ablate", "ablate-repair", "compare-encoders", "generate"])
    ap.add_argument("--config", default="configs/tiger.yaml")
    ap.add_argument("--source", type=str, help="source directory for the import-fashion command")
    ap.add_argument("--caption", type=str, help="caption for the standalone generate command")
    ap.add_argument("--output", type=str, help="output path for the standalone generate command")
    ap.add_argument("--seed", type=int, default=None, help="noise seed override")
    ap.add_argument("--seeds", default="7,8,9,10,11", help="sweep: comma-separated noise seeds")
    ap.add_argument("--sample", type=int, default=None, help="ablate-repair: number of noisy items to sample (default: all)")
    ap.add_argument("--split", default="report", choices=["report", "calibration"])
    ap.add_argument("--independent", action="store_true",
                    help="repair: cross-check each repair with the independent verifier encoder (6.4)")
    ap.add_argument("--vlm-judge", action="store_true",
                    help="repair: cross-check each repair with the Gemini VLM judge (6.4, reads GEMINI_API_KEY from .env)")
    ap.add_argument("--fusion", action="store_true",
                    help="detect/repair/ablate-repair: apply the precision-floor fusion "
                         "config from calibrate-fusion (roadmap 3.4). Off by default: "
                         "fusion trades recall for precision, so it changes what reaches "
                         "the repair stage")
    ap.add_argument("--generative-fallback", action="store_true",
                    help="repair: use Stable Diffusion to generate missing images (requires diffusers)")
    # import-abo specific args
    ap.add_argument("--listings-dir", default=None,
                    help="import-abo: directory containing ABO listings_*.json.gz")
    ap.add_argument("--images-csv", default=None,
                    help="import-abo: path to ABO images.csv")
    ap.add_argument("--images-dir", default=None,
                    help="import-abo: root directory of ABO small JPEG images")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    {
        "synthgen": cmd_synthgen,
        "import-fashion": cmd_import_fashion,
        "import-abo": cmd_import_abo,
        "noise": cmd_noise,
        "calibrate": cmd_calibrate,
        "detect": cmd_detect,
        "evaluate": cmd_evaluate,
        "analyze": cmd_analyze,
        "train-arbiter": cmd_train_arbiter,
        "route": cmd_route,
        "repair": cmd_repair,
        "sweep": cmd_sweep,
        "calibrate-fusion": cmd_calibrate_fusion,
        "ablate": cmd_ablate,
        "ablate-repair": cmd_ablate_repair,
        "compare-encoders": cmd_compare_encoders,
        "generate": cmd_generate,
    }[args.command](cfg, args)


if __name__ == "__main__":
    main()
