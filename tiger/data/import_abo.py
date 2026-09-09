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

from tiger.schema import Schema
from tiger import text_views

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category mapping: ABO product_type → TIGeR categories
# We map into the 4 NEW non-fashion categories added to schema.yaml.
# ---------------------------------------------------------------------------
CATEGORY_MAP: dict[str, str] = {
    # Electronics
    "CELLULAR_PHONE_CASE": "electronics",
    "HEADPHONES": "electronics",
    "SPEAKER": "electronics",
    "KEYBOARD": "electronics",
    "MOUSE": "electronics",
    "TABLET_CASE": "electronics",
    "LAPTOP_CASE": "electronics",
    "POWER_BANK": "electronics",
    "EARPHONES": "electronics",
    "SMARTWATCH": "electronics",
    # Furniture
    "TABLE": "furniture",
    "CHAIR": "furniture",
    "SOFA": "furniture",
    "SHELF": "furniture",
    "DESK": "furniture",
    "BED_FRAME": "furniture",
    "NIGHTSTAND": "furniture",
    "BOOKCASE": "furniture",
    "STOOL": "furniture",
    "CABINET": "furniture",
    # Kitchen
    "MUG": "kitchen",
    "WATER_BOTTLE": "kitchen",
    "PLATE": "kitchen",
    "BOWL": "kitchen",
    "CUTTING_BOARD": "kitchen",
    "PAN": "kitchen",
    "POT": "kitchen",
    "KETTLE": "kitchen",
    "STORAGE_CONTAINER": "kitchen",
    "COLANDER": "kitchen",
    # Home decor
    "LAMP": "home_decor",
    "CANDLE": "home_decor",
    "VASE": "home_decor",
    "PICTURE_FRAME": "home_decor",
    "CLOCK": "home_decor",
    "MIRROR": "home_decor",
    "THROW_PILLOW": "home_decor",
    "BLANKET": "home_decor",
    "RUG": "home_decor",
    "CURTAIN": "home_decor",
}

# Colors in ABO that we can map into our schema's color domain
# (schema already has standard colors; we only need to normalize ABO variants)
COLOR_ALIASES: dict[str, str] = {
    "silver": "gray",         # map metallic silver → gray (closest schema value)
    "gold": "yellow",         # gold → yellow
    "beige": "white",         # beige → white
    "tan": "brown",
    "navy": "blue",
    "navy blue": "blue",
    "teal": "green",
    "turquoise": "green",
    "ivory": "white",
    "cream": "white",
    "charcoal": "black",
    "burgundy": "red",
    "maroon": "red",
    "violet": "purple",
    "indigo": "blue",
    "magenta": "pink",
    "rose": "pink",
    "coral": "orange",
    "lime": "green",
    "olive": "green",
    "mint": "green",
    "lavender": "purple",
    "transparent": "white",
    "clear": "white",
    "multi": "multicolour",
    "multicolor": "multicolour",
    "multi-color": "multicolour",
    "multicolored": "multicolour",
    "assorted": "multicolour",
}


def _extract_color(raw_color: str | None, schema: Schema) -> str | None:
    """Normalise a raw ABO color string into a schema-valid value, or None."""
    if not raw_color:
        return None
    c = str(raw_color).strip().lower()
    c = COLOR_ALIASES.get(c, c)
    # Try schema normalise (handles aliases defined in schema.yaml)
    c = schema.normalize("color", c)
    if schema.in_domain("color", c):
        return c
    return None


def import_abo(
    listings_dir: Path,
    images_csv: Path,
    images_dir: Path,
    out_dir: Path,
    schema: Schema,
    max_items: int = 3000,
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
                if color is None:
                    continue  # color is required by schema
        
                attrs = {"color": color}
        
                # --- Material (optional, best-effort) ---
                raw_material = row.get("material", row.get("fabric_type", None))
                if raw_material:
                    mat = str(_extract_english_value(raw_material) or "").lower().strip()
                    mat = schema.normalize("material", mat)
                    if schema.in_domain("material", mat):
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
