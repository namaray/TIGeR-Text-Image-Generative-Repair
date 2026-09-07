"""Sieve: detection signals + contamination-robust thresholds.

Fixes vs the legacy run_sieve.py:

  F1  -- encoder inputs are short natural captions (tiger.text_views) with a
         hard tokeniser-length assertion; the k=v serialisation is never encoded.
  F3  -- thresholds tau_k are calibrated on the CLEAN calibration split only
         (quantile of clean similarities), never on the contaminated set.
  F4  -- calibration statistics are computed at unique-product level, and the
         per-category threshold falls back to global when the category has
         fewer than min_products_per_category unique products.
  F2  -- per-field contrastive probes (color/material/pattern): the image is
         scored against one caption per candidate value in Omega_j; the
         declared value must not lose to the best alternative by more than a
         z-scored margin. This is the instrument that can catch "red"->"blue".
  F15 -- probes are category-conditioned, prompt-ensembled, and fully
         instrumented (declared score, best-alternative, raw margin, z-margin
         all recorded per row).

Every signal keeps its own flag column so ablations and per-signal precision
(roadmap 3.4/3.6, 5.4) fall out for free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tiger import text_views
from tiger.encoders import ClipEncoder
from tiger.schema import Schema

SIGNAL_FLAGS = [
    "flag_missing_image",
    "flag_missing_text",
    "flag_low_sim",
    "flag_probe_color",
    "flag_probe_material",
    "flag_probe_pattern",
    "flag_text_out_of_domain",
    "flag_title_contradiction",
]


@dataclass
class SieveThresholds:
    """Locked calibration artefact; JSON-serialisable."""
    sim: dict            # category -> {"tau": float, "mean": float, "std": float, "n_products": int}
    sim_global: dict     # {"tau", "mean", "std", "n_products"}
    probes: dict         # field -> category -> {"mean": m, "std": s, "n_products": int}
    probes_global: dict  # field -> {"mean", "std", "n_products"}
    config: dict

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "SieveThresholds":
        return cls(**json.loads(s))


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, encoder: ClipEncoder, schema: Schema,
                    cfg: dict, root: Path) -> tuple[pd.DataFrame, dict]:
    """Compute similarity + probe signal columns.

    Returns (df_copy, arrays) where arrays holds image_emb, caption_emb and
    image_ok aligned to df_copy's positional index (kept out of the DataFrame
    so it stays parquet-serialisable).
    """
    df = df.reset_index(drop=True).copy()
    probes_cfg = cfg.get("sieve", {}).get("probes", {})
    probe_fields = [f for f in probes_cfg.get("fields", ["color"]) if f in schema.checkable_fields()]

    attrs_list = [text_views.parse_attrs(a) for a in df["attributes"]]
    captions = [text_views.full_caption(c, a) for c, a in zip(df["category"], attrs_list)]
    titles = [text_views.title_view(t) for t in df["title"]]

    # F1 guard: nothing we send to the encoder may exceed the token budget
    text_views.assert_token_budget(captions, context="(full captions)")
    text_views.assert_token_budget([t or " " for t in titles], context="(titles)")

    df["caption"] = captions

    # ---------- embeddings ----------
    img_paths = [str((root / p).resolve()) if p else "" for p in df["image_path"].astype(str)]
    image_emb = np.zeros((len(df), 1), dtype=np.float32)
    ok = np.zeros(len(df), dtype=bool)
    valid_paths = [p if p and Path(p).exists() else "" for p in img_paths]
    to_encode = [p for p in valid_paths if p]
    if to_encode:
        emb_all, ok_all = encoder.encode_images(to_encode)
        image_emb = np.zeros((len(df), emb_all.shape[1]), dtype=np.float32)
        j = 0
        for i, p in enumerate(valid_paths):
            if p:
                image_emb[i] = emb_all[j]
                ok[i] = ok_all[j]
                j += 1

    df["is_image_missing"] = df.get("is_image_missing", False)
    df["is_image_missing"] = df["is_image_missing"].astype(bool) | ~ok
    df["is_text_missing"] = df.get("is_text_missing", False)
    df["is_text_missing"] = df["is_text_missing"].astype(bool) | (df["title"].astype(str).str.strip() == "")

    cap_emb = encoder.encode_texts(captions)
    title_emb = encoder.encode_texts([t or " " for t in titles])

    sim_full = (cap_emb * image_emb).sum(axis=1)
    sim_title = (title_emb * image_emb).sum(axis=1)
    df["sim_full"] = np.where(ok, sim_full, np.nan)
    df["sim_title"] = np.where(ok, sim_title, np.nan)

    # ---------- per-field contrastive probes (3.1/3.2) ----------
    for fld in probe_fields:
        df[f"probe_{fld}_declared"] = np.nan
        df[f"probe_{fld}_best_alt"] = np.nan
        df[f"probe_{fld}_margin"] = np.nan
        df[f"probe_{fld}_pred"] = ""

    for fld in probe_fields:
        domain = schema.domain(fld)
        for cat in df["category"].astype(str).unique():
            # candidate caption embeddings once per (category, field), ensembled
            cand_embs = []
            for v in domain:
                templates = text_views.field_caption_templates(cat, fld, v)
                e = encoder.encode_texts(templates).mean(axis=0)
                e = e / (np.linalg.norm(e) + 1e-12)
                cand_embs.append(e)
            cand = np.stack(cand_embs)  # (K, D)

            m = (df["category"].astype(str) == cat) & ok
            idxs = df.index[m].tolist()
            if not idxs:
                continue
            scores = image_emb[idxs] @ cand.T  # (n, K)
            for pos, i in enumerate(idxs):
                declared = schema.normalize(fld, attrs_list[i].get(fld, "")) if attrs_list[i].get(fld) else ""
                pred_j = int(np.argmax(scores[pos]))
                df.at[i, f"probe_{fld}_pred"] = domain[pred_j]
                if declared and declared in domain:
                    dj = domain.index(declared)
                    ds = float(scores[pos, dj])
                    alt = float(np.max(np.delete(scores[pos], dj)))
                    df.at[i, f"probe_{fld}_declared"] = ds
                    df.at[i, f"probe_{fld}_best_alt"] = alt
                    df.at[i, f"probe_{fld}_margin"] = ds - alt

    # ---------- image-independent text checks (3.5) ----------
    ood, contra = [], []
    for i, (cat, attrs, title) in enumerate(zip(df["category"].astype(str), attrs_list, titles)):
        ood.append(bool(schema.validate_attrs(cat, attrs)))
        brand = str(attrs.get("brand", ""))
        title_color = _title_color(title, schema, brand)
        declared = schema.normalize("color", attrs.get("color", "")) if attrs.get("color") else ""
        contra.append(bool(title_color and declared and title_color != declared))
    df["flag_text_out_of_domain"] = ood
    df["flag_title_contradiction"] = contra

    encoder.save_cache()
    arrays = {"image_emb": image_emb, "caption_emb": cap_emb, "image_ok": ok}
    return df, arrays


def _title_color(title: str, schema: Schema, brand: str) -> str:
    """First colour word in the title, NORMALISED, ignoring brand names (D7).

    Scans surface forms rather than domain members, so "Navy Shirt" is
    recognised even though `navy` is an alias. Returns the canonical value so the
    caller can compare it against a normalised declared colour: returning the raw
    match made "Navy Shirt" + color=navy read as a contradiction, because
    "navy" != "blue".
    """
    import re

    masked = (title or "").lower()
    if brand:
        masked = masked.replace(brand.lower(), " ")
    for form in schema.surface_forms("color"):
        if schema.normalize("color", form) == "multicolour":
            continue
        if re.search(rf"\b{re.escape(form)}\b", masked):
            return schema.normalize("color", form)
    return ""


# ---------------------------------------------------------------------------
# calibration (clean split only) and application
# ---------------------------------------------------------------------------

def _product_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique product (first occurrence) -- effective sample units."""
    key = "product_id" if "product_id" in df.columns else "row_id"
    return df.drop_duplicates(subset=[key], keep="first")


def calibrate(df_signals_clean: pd.DataFrame, cfg: dict, schema: Schema) -> SieveThresholds:
    """Fit tau_k and probe z-stats on the CLEAN calibration split (F3/F4)."""
    scfg = cfg.get("sieve", {})
    q = float(scfg.get("quantile_q", 0.02))
    gq = float(scfg.get("global_quantile_q", q))
    min_prod = int(scfg.get("min_products_per_category", 8))
    probe_fields = [f for f in scfg.get("probes", {}).get("fields", []) if f in schema.checkable_fields()]

    d = _product_level(df_signals_clean)
    d = d[~d["sim_full"].isna()]

    def stats(vals: np.ndarray, quant: float) -> dict:
        return {
            "tau": float(np.quantile(vals, quant)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_products": int(len(vals)),
        }

    g_vals = d["sim_full"].astype(float).values
    sim_global = stats(g_vals, gq)

    sim = {}
    for cat, grp in d.groupby(d["category"].astype(str)):
        vals = grp["sim_full"].astype(float).values
        if len(vals) >= min_prod:
            sim[cat] = stats(vals, q)

    probes: dict[str, dict] = {}
    probes_global: dict[str, dict] = {}
    for fld in probe_fields:
        col = f"probe_{fld}_margin"
        dd = d[~d[col].isna()]
        if dd.empty:
            continue
        vals = dd[col].astype(float).values
        probes_global[fld] = {"mean": float(np.mean(vals)),
                              "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                              "n_products": int(len(vals))}
        probes[fld] = {}
        for cat, grp in dd.groupby(dd["category"].astype(str)):
            v = grp[col].astype(float).values
            if len(v) >= min_prod:
                probes[fld][cat] = {"mean": float(np.mean(v)),
                                    "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                    "n_products": int(len(v))}

    return SieveThresholds(
        sim=sim, sim_global=sim_global, probes=probes, probes_global=probes_global,
        config={"quantile_q": q, "global_quantile_q": gq,
                "min_products_per_category": min_prod,
                "z_margin": float(scfg.get("probes", {}).get("z_margin", 2.0)),
                "min_margin_raw": float(scfg.get("probes", {}).get("min_margin_raw", 0.005)),
                "probe_fields": probe_fields},
    )


def apply_thresholds(df_signals: pd.DataFrame, thr: SieveThresholds,
                     enabled_signals: list[str] | None = None,
                     fusion=None) -> pd.DataFrame:
    """Apply LOCKED thresholds; emit per-signal flags + fused `flagged`.

    `fusion` (tiger.fusion.FusionConfig) optionally supplies per-signal z-margins
    and quarantine flags calibrated to a precision floor (roadmap 3.4). Without
    it every probe uses the single default z-margin from calibration.
    """
    df = df_signals.copy()
    z_margin = float(thr.config.get("z_margin", 2.0))
    probe_fields = thr.config.get("probe_fields", [])

    cats = df["category"].astype(str)
    tau = cats.map(lambda c: thr.sim.get(c, thr.sim_global)["tau"])
    mean = cats.map(lambda c: thr.sim.get(c, thr.sim_global)["mean"])
    std = cats.map(lambda c: thr.sim.get(c, thr.sim_global)["std"]).replace(0.0, np.nan)
    df["sieve_tau"] = tau
    df["sim_z"] = (df["sim_full"] - mean) / std
    df["threshold_source"] = cats.map(lambda c: "category" if c in thr.sim else "global")

    df["flag_missing_image"] = df["is_image_missing"].astype(bool)
    df["flag_missing_text"] = df["is_text_missing"].astype(bool)
    df["flag_low_sim"] = (~df["sim_full"].isna()) & (df["sim_full"] < tau)

    for fld in probe_fields:
        col = f"probe_{fld}_margin"
        flag_col = f"flag_probe_{fld}"
        if col not in df.columns:
            df[flag_col] = False
            continue
        pm = cats.map(lambda c: thr.probes.get(fld, {}).get(c, thr.probes_global.get(fld, {"mean": 0.0}))["mean"])
        ps = cats.map(lambda c: thr.probes.get(fld, {}).get(c, thr.probes_global.get(fld, {"std": 0.0}))["std"])
        ps = ps.replace(0.0, np.nan)
        z = (df[col] - pm) / ps
        df[f"probe_{fld}_z"] = z
        margin = fusion.z_margin(flag_col, z_margin) if fusion is not None else z_margin
        df[flag_col] = (~df[col].isna()) & (z <= -margin) & (df[col] < 0)
        if fusion is not None and fusion.is_quarantined(flag_col):
            df[flag_col] = False  # signal failed the precision floor (roadmap 3.6)

    for c in SIGNAL_FLAGS:
        if c not in df.columns:
            df[c] = False

    active = enabled_signals if enabled_signals is not None else SIGNAL_FLAGS
    if fusion is not None and enabled_signals is None:
        active = [s for s in SIGNAL_FLAGS if not fusion.is_quarantined(s)]
    df["flagged"] = False
    for c in active:
        df["flagged"] |= df[c].astype(bool)

    def reason(r) -> str:
        for c in active:
            if bool(r[c]):
                return c.replace("flag_", "")
        return "ok"

    df["flag_reason"] = df.apply(reason, axis=1)
    return df
