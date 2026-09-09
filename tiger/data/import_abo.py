"""ABO Dataset Adapter: Converts Amazon Berkeley Objects (ABO) Kaggle export to TIGeR parquet format.

Expected Kaggle dataset inputs (add both to your notebook):
  1. ABO Metadata dataset  → contains two CSVs:
       listings.csv  (item_id, item_name, product_type, color, main_image_id, ...)
       images.csv    (image_id, path, height, width)
  2. ABO Images dataset (abo-images-small) → contains the actual JPEG files

Usage (in tiger.ipynb on Kaggle):
    !python -m tiger.cli import-abo \\
        --listings /kaggle/input/abo-metadata/listings.csv \\
        --images-csv /kaggle/input/abo-metadata/images.csv \\
        --images-dir /kaggle/input/abo-images-small/images/small
"""

import gzip
import json
import logging
import random
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from tiger.data import abo_vocab
from tiger.schema import Schema
from tiger import text_views

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category mapping: ABO product_type → TIGeR categories
# We map into the 4 NEW non-fashion categories added to schema.yaml.
# ---------------------------------------------------------------------------
# Two verticals of the ABO catalogue, replacing the Kaggle Myntra fashion set.
# Each ABO product_type maps to its own TIGeR category rather than to a coarse
# vertical: per-category tau thresholds and the T2V candidate pool are both
# category-scoped, so a chair should only ever be repaired with another chair.
# The vertical grouping below is for reporting and the cross-domain contrast.
#
# Chosen on measured attribute coverage over the local ABO release, not on
# intuition. Footwear was rejected despite being the largest fashion-like block:
# 1,893 distinct colour strings and 8-28% material coverage would have left
# material_flip unmeasurable, which is already the weakest signal in the paper.
VERTICAL_A_FURNISHING = {
    "CHAIR": "chair", "SOFA": "sofa", "TABLE": "table", "OTTOMAN": "ottoman",
    "STOOL_SEATING": "stool", "RUG": "rug", "LAMP": "lamp",
    "LIGHT_FIXTURE": "light_fixture", "WALL_ART": "wall_art",
}

VERTICAL_B_ACCESSORIES = {
    "FINERING": "ring", "FINENECKLACEBRACELETANKLET": "necklace",
    "FINEEARRING": "earring", "HANDBAG": "handbag",
    "SUITCASE": "suitcase", "HAT": "hat",
}

CATEGORY_MAP: dict[str, str] = {**VERTICAL_A_FURNISHING, **VERTICAL_B_ACCESSORIES}

VERTICALS: dict[str, set[str]] = {
    "furnishing": set(VERTICAL_A_FURNISHING.values()),
    "accessories": set(VERTICAL_B_ACCESSORIES.values()),
}

# CELLULAR_PHONE_CASE is deliberately absent: 64,853 listings, 44% of the whole
# catalogue. Including it unsubsampled would swamp every other category. It is
# better used as a dedicated class-imbalance stress test (reviewer_defense.md
# Attack 9), not as background.

def _extract_color(raw_color, schema: Schema) -> str | None:
    """Free-text ABO colour -> Omega_color, or None when unresolvable.

    Delegates to tiger.data.abo_vocab, which handles modifiers ("light grey"),
    other languages ("blanco", "silber"), and explicit non-values ("no aplica").
    Resolves 84.1% of colour instances in the two furnishing verticals; the
    remainder is a long tail that returns None rather than a guess.
    """
    domain = {schema.normalize("color", v) for v in schema.domain("color")}
    return abo_vocab.normalize_color(raw_color, domain)


def _extract_material(raw_material, schema: Schema) -> str | None:
    """Free-text ABO material -> Omega_material, or None. Resolves 82.0%."""
    domain = {schema.normalize("material", v) for v in schema.domain("material")}
    return abo_vocab.normalize_material(raw_material, domain)


def import_abo(
    listings_dir: Path,
    images_csv: Path,
    images_dir: Path,
    out_dir: Path,
    schema: Schema,
    max_items: int = 3000,
    max_per_category: int | None = 1200,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Parse ABO Kaggle JSON lines and produce products.parquet for the TIGeR pipeline.

    Args:
        listings_dir: Directory containing ABO listings_*.json files.
        images_csv:   Path to ABO images.csv (image_id → path mapping).
        images_dir:   Root directory of the ABO small JPEG images.
        out_dir:      Where to write products.parquet + meta.json.
        schema:       Loaded TIGeR schema for validation.
        max_items:    Maximum number of products to import.
        seed:         Random seed for calibration/report split.
    """
    rng = random.Random(seed)

    log.info("Loading ABO image map from %s ...", images_csv)
    images = pd.read_csv(images_csv, low_memory=False)

    # Build image_id → relative file path lookup
    img_path_map: dict[str, Path] = {}
    for _, row in images.iterrows():
        iid = str(row.get("image_id", "")).strip()
        rel = str(row.get("path", "")).strip()
        if iid and rel:
            # 1. Try direct join
            full = (images_dir / rel).resolve()
            
            # 2. If it doesn't exist, try stripping redundant "images/small/" prefix
            if not full.exists():
                if rel.startswith("images/small/"):
                    full = (images_dir / rel[len("images/small/"):]).resolve()
                elif rel.startswith("small/"):
                    full = (images_dir / rel[len("small/"):]).resolve()
                    
            if full.exists():
                img_path_map[iid] = full

    log.info("Image path map built: %d valid images found.", len(img_path_map))

    rows = []
    seen_products: set[str] = set()
    # Per-category cap. ABO is dominated by a few product types; uncapped, one
    # category sets the global tau and fills the T2V candidate pool for every
    # other. None disables the cap.
    per_cat: dict[str, int] = {}

    # Accept both layouts. The official ABO archive (abo-listings.tar from
    # s3://amazon-berkeley-objects/) ships listings_*.json.gz; the Kaggle mirror
    # had already decompressed them. H2 fixed a gzip-only reader by making it
    # plain-only, which silently swapped one half of the problem for the other
    # and made the canonical source unusable. Handle both.
    json_files = sorted(listings_dir.glob("listings_*.json")) + \
                 sorted(listings_dir.glob("listings_*.json.gz"))
    if not json_files:
        raise FileNotFoundError(
            f"No listings_*.json or listings_*.json.gz files found in {listings_dir}")

    log.info("Parsing ABO JSON listings from %d files...", len(json_files))
    
    for json_file in json_files:
        if len(rows) >= max_items:
            break
            
        opener = gzip.open if json_file.suffix == ".gz" else open
        with opener(json_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if len(rows) >= max_items:
                    break
                    
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # --- Category ---
                raw_pt = row.get("product_type", "")
                pt = _extract_english_value(raw_pt)
                if not pt:
                    continue
                product_type = pt.strip().upper()
                category = CATEGORY_MAP.get(product_type)
                if category is None:
                    continue  # Skip unmapped product types
                if max_per_category is not None and per_cat.get(category, 0) >= max_per_category:
                    continue
        
                # --- Title (English) ---
                # ABO may store item_name as a JSON array of {"language_tag": ..., "value": ...}
                raw_name = row.get("item_name", "")
                title = _extract_english_value(raw_name)
                if not title:
                    continue
        
                # --- Product ID ---
                product_id = str(row.get("item_id", "")).strip()
                if not product_id or product_id in seen_products:
                    continue
        
                # --- Image ---
                main_image_id = str(row.get("main_image_id", "")).strip()
                img_path = img_path_map.get(main_image_id)
                if img_path is None:
                    continue  # Skip products whose image is not in the small archive
        
                # --- Color ---
                raw_color = row.get("color", row.get("colors", None))
                color = _extract_color(_extract_english_value(raw_color), schema)

                # Previously: `if color is None: continue`. That restricted the
                # corpus to products whose colour string happened to resolve,
                # biasing every colour-repair number measured on it -- and it
                # meant `attribute_drop` noise was the only way a row could ever
                # lack a colour. Colour is category-scoped in schema.yaml, so a
                # chair without one is a valid record and a realistic one.
                attrs = {}
                if color is not None:
                    attrs["color"] = color
        
                # --- Material (optional, best-effort) ---
                raw_material = row.get("material", row.get("fabric_type", None))
                mat = _extract_material(_extract_english_value(raw_material), schema)
                if mat is not None:
                    attrs["material"] = mat
        
                seen_products.add(product_id)
                rows.append({
                    "row_id": product_id,
                    "product_id": product_id,
                    "title": title,
                    "category": category,
                    "attributes": json.dumps(attrs, ensure_ascii=False),
                    "canonical_text": text_views.canonical_text(title, category, attrs),
                    "image_path": str(img_path),
                    "is_image_missing": False,
                    "is_text_missing": False,
                })
                per_cat[category] = per_cat.get(category, 0) + 1

    if not rows:
        raise ValueError(
            "No valid ABO products matching our target categories were found. "
            "Check that listings_csv, images_csv, and images_dir are all correctly set."
        )

    df = pd.DataFrame(rows)

    # Calibration / report split (50/50 by product)
    products = df["product_id"].tolist()
    rng.shuffle(products)
    n_cal = int(len(products) * 0.5)
    cal_set = set(products[:n_cal])
    df["split"] = df["product_id"].map(lambda p: "calibration" if p in cal_set else "report")

    # Write output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "products.parquet"
    df.to_parquet(out_file, index=False)
    log.info("Imported %d ABO products → %s", len(df), out_file)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_listings_dir": str(listings_dir),
        "source_images": str(images_csv),
        "seed": seed,
        "n_products": len(df),
        "by_category": df["category"].value_counts().to_dict(),
        "by_split": df["split"].value_counts().to_dict(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return df


def _extract_english_value(raw) -> str | None:
    """
    ABO encodes multilingual fields as either a plain string or a list of dicts:
      [{"language_tag": "en_US", "value": "Blue Mug"}, ...]
    This function extracts the English value regardless of format.
    """
    if raw is None or (isinstance(raw, float)):
        return None
        
    items = raw
    # If it's a raw string that looks like a JSON array, try parsing it
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            pass

    # If we have a list of dictionaries, extract the English one
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                lang = str(item.get("language_tag", "")).lower()
                if lang.startswith("en"):
                    return str(item.get("value", "")).strip() or None
        # Fallback: return first item's value regardless of language
        if items and isinstance(items[0], dict):
            return str(items[0].get("value", "")).strip() or None
            
    # If it's not a list (or failed to parse as one), just return it as a string
    s = str(raw).strip()
    if not s or s == "nan":
        return None
    return s
