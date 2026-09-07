from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tiger.colors import estimate_dominant_color


def draw(tmp_path: Path, name: str, paint) -> Path:
    im = Image.new("RGB", (128, 128), (245, 245, 245))
    d = ImageDraw.Draw(im)
    paint(d)
    p = tmp_path / f"{name}.png"
    im.save(p)
    return p


def test_solid_red(tmp_path):
    p = draw(tmp_path, "red", lambda d: d.rectangle([20, 20, 108, 108], fill=(200, 35, 35)))
    assert estimate_dominant_color(p).top == "red"


def test_striped_red_white_not_pink(tmp_path):
    """F6 regression: mean-RGB called red/white stripes pink; mode must not."""
    def paint(d):
        for x in range(20, 108, 16):
            d.rectangle([x, 20, x + 8, 108], fill=(200, 35, 35))
            d.rectangle([x + 8, 20, x + 16, 108], fill=(250, 250, 250))
    p = draw(tmp_path, "stripes", paint)
    est = estimate_dominant_color(p)
    assert est.top in ("red", "multicolour")
    assert est.top != "pink"


def test_white_background_does_not_dominate(tmp_path):
    # small blue object on a big white studio background
    p = draw(tmp_path, "smallblue", lambda d: d.rectangle([44, 44, 84, 84], fill=(45, 75, 200)))
    assert estimate_dominant_color(p).top == "blue"


def test_black_product(tmp_path):
    p = draw(tmp_path, "black", lambda d: d.rectangle([20, 20, 108, 108], fill=(20, 20, 22)))
    assert estimate_dominant_color(p).top == "black"


def test_white_product(tmp_path):
    p = draw(tmp_path, "white", lambda d: d.rectangle([10, 10, 118, 118], fill=(250, 250, 250)))
    assert estimate_dominant_color(p).top == "white"


def test_multicolour(tmp_path):
    def paint(d):
        d.rectangle([20, 20, 64, 108], fill=(200, 35, 35))
        d.rectangle([64, 20, 108, 108], fill=(45, 75, 200))
    p = draw(tmp_path, "redblue", paint)
    est = estimate_dominant_color(p)
    assert est.top == "multicolour"
    names = {n for n, _ in est.top2}
    assert names == {"red", "blue"}
