"""Solver: repair planning + structure-safe patch application (roadmap 1.5/2.7).

The patch PRIMITIVE (apply_attr_patch, F10) validates and applies a single
attribute patch. Planning (plan_repair, roadmap 2.7) turns an Arbiter Route plus
Analyzer Evidence into a concrete, cost-minimal repair payload:

  - V2T (E1, text wrong): enumerate candidate single-field patches from the
    evidence (the suspect field from Eq. 18 plus any fired probe), take the
    probe-winning value (the image's own best match), and keep the minimal
    valid one. Colour uses the deterministic HSV pixel estimate when confident.
  - T2V (E2/E3, image wrong): pick the catalogue image that best matches THIS
    row's text from a candidate pool that EXCLUDES the row's own product
    (finding F14: the pristine original is never handed back, so a passing
    repair means a genuinely better substitute, not the trivial inverse).

Guarantees of the primitive (unchanged): attribute-dict edits only, Omega_j and
constraint-set C validation before mutation, brand-safe title regeneration,
canonical_text rebuilt from structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tiger import text_views
from tiger.schema import Schema


@dataclass
class PatchResult:
    applied: bool
    title: str
    attrs: dict
    canonical_text: str
    changed_fields: list = field(default_factory=list)
    refusal_reason: str = ""
    escalate: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def apply_attr_patch(title: str, category: str, attrs: dict, patch: dict,
                     schema: Schema) -> PatchResult:
    """Apply {field: new_value} to the attributes dict, structure-safely.

    Refuses (escalate=True, row untouched) when any value is out of domain or
    the patched record violates the constraint set C.
    """
    attrs = dict(attrs)
    title = str(title or "")

    if not patch:
        return PatchResult(False, title, attrs,
                           text_views.canonical_text(title, category, attrs),
                           refusal_reason="empty_patch")

    # ---- validate BEFORE mutating anything (Omega_j gate) ----
    normalized: dict[str, str] = {}
    for fld, value in patch.items():
        if fld not in schema.attributes:
            return PatchResult(False, title, attrs,
                               text_views.canonical_text(title, category, attrs),
                               refusal_reason=f"unknown_field:{fld}", escalate=True)
        if not schema.in_domain(fld, value):
            return PatchResult(False, title, attrs,
                               text_views.canonical_text(title, category, attrs),
                               refusal_reason=f"out_of_domain:{fld}={value!r}", escalate=True)
        normalized[fld] = schema.normalize(fld, value)

    candidate = dict(attrs)
    candidate.update(normalized)
    violations = schema.validate_attrs(category, candidate)
    if violations:
        return PatchResult(False, title, attrs,
                           text_views.canonical_text(title, category, attrs),
                           refusal_reason="constraint_violation:" +
                                          "; ".join(v.message for v in violations),
                           escalate=True)

    # ---- apply ----
    changed = []
    brands = {str(attrs.get("brand", ""))}
    new_title = title
    for fld, new_value in normalized.items():
        old_value = schema.normalize(fld, attrs.get(fld, "")) if attrs.get(fld) else ""
        if old_value == new_value:
            continue
        if fld == "color":
            # regenerate the title's colour mention rather than word-replacing
            # arbitrary text: replace only a colour word occurring OUTSIDE
            # brand names; if none exists, leave the title alone.
            present = text_views.find_color_word(new_title, schema.domain("color"), brands)
            if present:
                new_title = text_views.replace_color_word_safe(new_title, present, new_value, brands)
        changed.append(fld)

    attrs.update(normalized)
    return PatchResult(True, new_title, attrs,
                       text_views.canonical_text(new_title, category, attrs),
                       changed_fields=changed)


# ---------------------------------------------------------------------------
# repair planning (roadmap 2.7)
# ---------------------------------------------------------------------------

@dataclass
class RepairPlan:
    row_id: str
    direction: str                       # V2T | T2V | NONE
    patch: dict = field(default_factory=dict)      # V2T: {field: value}
    candidate_image_path: str = ""       # T2V: catalogue image to swap in
    candidate_product_id: str = ""
    cost: float = 0.0                    # edit cost (fields changed / tier weight)
    plannable: bool = True
    notes: str = ""
    # --- V2T estimator attribution (diagnostics only; no effect on behaviour) ---
    # Restoration accuracy is determined by which value the operator writes, not
    # by which field the Arbiter routed to. These record both candidate
    # estimators and which one supplied the value, so operator error can be
    # attributed to the pixel path or the CLIP path independently of routing.
    value_source: str = ""               # "pixel" | "probe" | ""
    pixel_value: str = ""                # HSV dominant-colour estimate
    pixel_conf: float | None = None      # winning colour's pixel share (NOT calibrated)
    probe_value: str = ""                # CLIP per-field probe argmax
    estimators_agree: bool | None = None # pixel == probe (both non-empty)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class CandidatePool:
    """Images currently referenced by rows, indexed for T2V retrieval.

    The pool deliberately holds each product's *in-use* image, so a swapped
    row's true original (referenced by no one) is absent -- the F14 held-out
    protocol, applied to the repair operator itself rather than only to eval.
    """

    def __init__(self, image_emb: np.ndarray, product_ids: list[str],
                 image_paths: list[str], ok: np.ndarray):
        self.image_emb = image_emb
        self.product_ids = np.asarray(product_ids, dtype=object)
        self.image_paths = np.asarray(image_paths, dtype=object)
        self.ok = np.asarray(ok, dtype=bool)

    def best_for_text(self, caption_emb: np.ndarray, exclude_product: str,
                      category_ids: np.ndarray | None = None,
                      category: str | None = None) -> tuple[int, float] | None:
        sims = self.image_emb @ caption_emb
        mask = self.ok & (self.product_ids != exclude_product)
        if category_ids is not None and category is not None:
            mask &= (category_ids == category)
        if not mask.any():
            return None
        sims = np.where(mask, sims, -np.inf)
        j = int(np.argmax(sims))
        return j, float(sims[j])


def _corrected_value(field: str, ev: dict) -> tuple[str, dict]:
    """The image-supported replacement value for a suspect text field.

    Returns ``(value, diagnostics)``. The chosen value is unchanged from the
    original single-return behaviour; the diagnostics additionally record the
    losing estimator so that operator accuracy can be decomposed after a run
    (see RepairPlan's estimator-attribution fields).
    """
    probe_value = str((ev.get("probes") or {}).get(field, {}).get("pred", "") or "")
    pixel_value, pixel_conf = "", None

    if field == "color":
        pc = ev.get("pixel_color")
        conf = ev.get("pixel_color_confidence")
        pixel_value = str(pc or "")
        pixel_conf = float(conf) if conf is not None else None

    agree = bool(pixel_value and probe_value and pixel_value == probe_value)
    diag = {"pixel_value": pixel_value, "pixel_conf": pixel_conf,
            "probe_value": probe_value, "estimators_agree": agree}

    # deterministic pixel estimate wins when confident; else CLIP probe
    if (field == "color" and pixel_value
            and pixel_value not in ("unknown", "multicolour")
            and pixel_conf is not None and pixel_conf >= 0.55):
        return pixel_value, {**diag, "value_source": "pixel"}

    return probe_value, {**diag, "value_source": "probe"}


def plan_repair(ev: dict, route, sieve_row: dict, pool: CandidatePool,
                cat_ids: np.ndarray, caption_emb: np.ndarray, schema: Schema,
                same_category_only: bool = True, generator=None, root_path=None) -> RepairPlan:
    """Build a concrete repair payload from evidence + route."""
    row_id = str(ev.get("row_id", ""))
    category = str(ev.get("category", ""))
    attrs = text_views.parse_attrs(sieve_row.get("attributes", {}))

    if route.direction in ("V2T",):
        # candidate suspect fields: Eq.18 top + any fired probe, cost-ranked
        suspects: list[str] = []
        top = ev.get("loo_top_field")
        if top:
            suspects.append(top)
        for f in (ev.get("probes") or {}):
            z = (ev["probes"][f] or {}).get("z")
            if z is not None and z <= -2.0 and f not in suspects:
                suspects.append(f)
        for f in suspects:
            val, diag = _corrected_value(f, ev)
            if not val or not schema.in_domain(f, val):
                continue
            if schema.normalize(f, val) == (schema.normalize(f, attrs.get(f, "")) if attrs.get(f) else ""):
                continue  # no-op; the image already agrees with the text
            return RepairPlan(row_id, "V2T", patch={f: val}, cost=1.0,
                              notes=f"V2T single-field patch {f}={val} (suspect via LOO/probe)",
                              **diag)
        return RepairPlan(row_id, "V2T", plannable=False,
                          notes="no valid single-field patch from evidence")

    if route.direction in ("T2V", "BOTH"):
        res = pool.best_for_text(caption_emb, exclude_product=str(ev.get("product_id", "")),
                                 category_ids=cat_ids if same_category_only else None,
                                 category=category if same_category_only else None)
        if res is None:
            if generator is not None and root_path is not None:
                # Generative Fallback
                caption = str(sieve_row.get("canonical_text", ""))
                if not caption:
                    caption = str(sieve_row.get("title", ""))
                gen_path = root_path / "data" / "sample" / "images" / "generated" / f"{row_id}.jpg"
                generator.generate(caption, gen_path, category=category, attrs=attrs)
                rel_path = f"data/sample/images/generated/{row_id}.jpg"
                
                return RepairPlan(row_id, "T2V", candidate_product_id="GENERATED",
                                  candidate_image_path=rel_path, cost=2.0,
                                  notes="T2V fallback: synthesized new image from text")
            else:
                return RepairPlan(row_id, "T2V", plannable=False, notes="no candidate image in pool")
        j, _ = res
        return RepairPlan(row_id, "T2V",
                          candidate_image_path=str(pool.image_paths[j]),
                          candidate_product_id=str(pool.product_ids[j]),
                          cost=2.0,
                          notes=f"T2V replace image from product {pool.product_ids[j]} "
                                f"(own original held out, F14)")

    return RepairPlan(row_id, "NONE", plannable=False, notes=f"direction {route.direction} not repairable")
