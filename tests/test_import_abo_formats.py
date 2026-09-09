"""The ABO adapter must read the canonical archive, not only the Kaggle mirror.

The official dataset (s3://amazon-berkeley-objects/, abo-listings.tar) ships
listings_*.json.gz. The Kaggle mirror `khyeh0719/amazon-berkeley-objects-small`
had already decompressed them, and H2 "fixed" a gzip-only reader by making it
plain-only -- which swapped one half of the problem for the other and left the
citable source unreadable.
"""

import gzip
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from tiger.data.import_abo import import_abo
from tiger.schema import load_schema

RECORD = {
    "item_id": "B01",
    "product_type": [{"language_tag": "en_US", "value": "CHAIR"}],
    "item_name": [{"language_tag": "en_US", "value": "Oak Dining Chair"}],
    "color": [{"language_tag": "en_US", "value": "brown"}],
    "material": [{"language_tag": "en_US", "value": "wood"}],
    "main_image_id": "IMG1",
}


def _fixture(extension: str) -> dict:
    d = Path(tempfile.mkdtemp())
    (d / "meta").mkdir()
    payload = json.dumps(RECORD) + "\n"
    target = d / "meta" / f"listings_0{extension}"
    if extension.endswith(".gz"):
        target.write_bytes(gzip.compress(payload.encode()))
    else:
        target.write_text(payload)

    images = d / "img"
    (images / "aa").mkdir(parents=True)
    (images / "aa" / "1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)
    pd.DataFrame([{"image_id": "IMG1", "path": "aa/1.jpg", "height": 256, "width": 256}]) \
        .to_csv(d / "images.csv", index=False)

    return {"listings_dir": d / "meta", "images_csv": d / "images.csv",
            "images_dir": images, "out_dir": d / "out"}


@pytest.mark.parametrize("extension", [".json", ".json.gz"])
def test_both_listing_layouts_import(extension):
    """.json is the Kaggle mirror; .json.gz is the official archive."""
    df = import_abo(schema=load_schema("configs/schema.yaml"), max_items=10, seed=7,
                    **_fixture(extension))
    assert len(df) == 1
    assert df.iloc[0]["category"] == "furniture"


def test_missing_listings_names_both_layouts():
    """The error must name both layouts, so a wrong path is diagnosable.

    Uses a complete fixture and then empties the listings directory: the image
    map is loaded before the listings are globbed, so omitting images.csv would
    raise pandas' FileNotFoundError instead and the assertion would pass for
    the wrong reason.
    """
    fx = _fixture(".json")
    for stale in fx["listings_dir"].glob("listings_*"):
        stale.unlink()

    with pytest.raises(FileNotFoundError, match=r"listings_\*\.json\.gz"):
        import_abo(schema=load_schema("configs/schema.yaml"), **fx)
