"""Token-budgeted natural-caption renderings (roadmap 1.2, findings F1/F2).

The old canonical rendering ("{title}. Color: c. Category: k. Attributes: k=v, ...")
is a database serialisation: out of distribution for CLIP's text encoder and long
enough that the attribute tail falls past the 77-token truncation point (F1).

This module renders short natural-language views instead:

  - full caption  : "a photo of a red cotton shirt with a striped pattern"
  - title view    : the raw title (already short)
  - field caption : "a photo of a {value} {category-singular}" per checkable field

`canonical_text` remains a storage format built from the attributes dict; it is
never fed to CLIP directly.  `assert_token_budget` provides the ingestion-time
tokeniser-length check the roadmap requires.
"""

from __future__ import annotations

import json
from typing import Any

# Finding D8: every category needs a real singular noun. Anything falling through
# to rstrip("s") produced captions like "a photo of a red home_decor" -- an
# underscore token and a non-noun, fed to CLIP as the entire prompt ensemble, so
# every probe margin and LOO delta derived from it was measured against a
# degraded prompt.
CATEGORY_SINGULAR = {
    # synthetic fashion catalogue
    "shirts": "shirt",
    "shoes": "shoe",
    "bags": "bag",
    "hats": "hat",
    # ABO vertical A -- furnishing
    "chair": "chair",
    "sofa": "sofa",
    "table": "table",
    "ottoman": "ottoman",
    "stool": "stool",
    "rug": "rug",
    "lamp": "lamp",
    "light_fixture": "light fixture",
    "wall_art": "piece of wall art",
    # ABO vertical B -- accessories
    "ring": "ring",
    "necklace": "necklace",
    "earring": "earring",
    "handbag": "handbag",
    "suitcase": "suitcase",
    "hat": "hat",
}

CLIP_TOKEN_LIMIT = 77


def singular(category: str) -> str:
    c = str(category).strip().lower()
    return CATEGORY_SINGULAR.get(c, c.rstrip("s") or "item")


def parse_attrs(val: Any) -> dict:
    if isinstance(val, dict):
        return dict(val)
    if val is None:
        return {}
    s = str(val).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def full_caption(category: str, attrs: dict) -> str:
    """Short natural caption for the global sieve view.

    Deliberately excludes the title (see title_view) and low-value fields so the
    caption stays well inside the token budget.
    """
    a = {str(k).lower(): str(v).strip().lower() for k, v in attrs.items() if v not in (None, "", [])}
    noun = singular(category)

    parts = ["a photo of a"]
    if a.get("color"):
        parts.append(a["color"])
    if a.get("material"):
        parts.append(a["material"])
    parts.append(noun)

    tail = []
    if a.get("pattern") and a["pattern"] != "solid":
        tail.append(f"with a {a['pattern']} pattern")
    if a.get("brand"):
        tail.append(f"by {attrs.get('brand')}")

    return " ".join(parts + tail)


def full_caption_paraphrase(category: str, attrs: dict) -> str:
    """A meaning-preserving rewording of full_caption.

    Used only to measure the caption-rewording noise floor for Eq. 29 epsilon
    (tiger.verify): the change in image-text similarity induced by rephrasing a
    caption that still describes the same product bounds how large a repair's
    improvement must be to count as real rather than wording wobble.
    """
    a = {str(k).lower(): str(v).strip().lower() for k, v in attrs.items() if v not in (None, "", [])}
    noun = singular(category)

    descriptors = []
    if a.get("material"):
        descriptors.append(a["material"])
    if a.get("color"):
        descriptors.append(a["color"])

    head = f"a product photo of a {noun}"
    if descriptors:
        head += " in " + " ".join(descriptors)

    tail = []
    if a.get("pattern") and a["pattern"] != "solid":
        tail.append(f"showing a {a['pattern']} pattern")
    if a.get("brand"):
        tail.append(f"from {attrs.get('brand')}")

    return " ".join([head] + tail)


def title_view(title: str) -> str:
    return str(title or "").strip()


def field_caption(category: str, field: str, value: str) -> str:
    """Per-field probe caption: one short caption per candidate value."""
    noun = singular(category)
    v = str(value).strip().lower()
    if field == "color":
        return f"a photo of a {v} {noun}"
    if field == "material":
        return f"a photo of a {noun} made of {v}"
    if field == "pattern":
        return f"a photo of a {noun} with a {v} pattern" if v != "solid" else f"a photo of a plain solid-colour {noun}"
    return f"a photo of a {noun}, {field} {v}"


def field_caption_templates(category: str, field: str, value: str) -> list[str]:
    """Prompt ensemble (roadmap 3.2): 3-5 templates averaged in embedding space."""
    noun = singular(category)
    v = str(value).strip().lower()
    if field == "color":
        return [
            f"a photo of a {v} {noun}",
            f"a {v} {noun} on a white background",
            f"a product photo of a {v} {noun}",
            f"an image of a {v} coloured {noun}",
        ]
    if field == "material":
        return [
            f"a photo of a {noun} made of {v}",
            f"a {v} {noun} on a white background",
            f"a product photo of a {v} {noun}",
        ]
    if field == "pattern":
        if v == "solid":
            return [
                f"a photo of a plain solid-colour {noun}",
                f"a {noun} in a single solid colour",
                f"a product photo of a plain {noun}",
            ]
        return [
            f"a photo of a {noun} with a {v} pattern",
            f"a {v} {noun} on a white background",
            f"a product photo of a {v} patterned {noun}",
        ]
    return [field_caption(category, field, value)]


def canonical_text(title: str, category: str, attrs: dict) -> str:
    """Storage/serialisation format only -- never encoded by CLIP."""
    items = sorted((str(k), str(v)) for k, v in attrs.items())
    attrs_for_text = ", ".join(f"{k}={v}" for k, v in items)
    return f"{title}. Category: {category}. Attributes: {attrs_for_text}."


# ---------- brand-safe colour-word replacement (F10) ----------

def replace_color_word_safe(title: str, old: str, new: str, brands: set[str]) -> str:
    """Replace a colour word in a title without corrupting brand/pattern names.

    Brand substrings (e.g. "Red Wing Supply") are masked before replacement so
    "Red Wing Supply Red Cotton Shirt" -> "Red Wing Supply Blue Cotton Shirt".
    """
    import re

    masked = title
    placeholders = {}
    for i, b in enumerate(sorted({b for b in brands if b}, key=len, reverse=True)):
        if b.lower() in masked.lower():
            ph = f"\x00B{i}\x00"
            placeholders[ph] = None
            # preserve the original casing of the brand occurrence
            m = re.search(re.escape(b), masked, flags=re.IGNORECASE)
            placeholders[ph] = masked[m.start():m.end()]
            masked = masked[:m.start()] + ph + masked[m.end():]
    def _match_case(m: "re.Match") -> str:
        src = m.group(0)
        if src.isupper():
            return new.upper()
        if src[:1].isupper():
            return new.capitalize()
        return new

    masked = re.sub(rf"\b{re.escape(old)}\b", _match_case, masked, flags=re.IGNORECASE)
    for ph, original in placeholders.items():
        masked = masked.replace(ph, original)
    return masked


def find_color_word(title: str, colors: list[str], brands: set[str]) -> str | None:
    """First colour word in the title, ignoring colour words inside brand names."""
    import re

    masked = (title or "").lower()
    for b in brands:
        if b:
            masked = masked.replace(b.lower(), " ")
    for c in colors:
        if re.search(rf"\b{re.escape(c)}\b", masked):
            return c
    return None


# ---------- token budget assertion (roadmap 1.2) ----------

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import CLIPTokenizerFast

        _tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    return _tokenizer


def token_length(text: str) -> int:
    return len(_get_tokenizer()(text)["input_ids"])


def assert_token_budget(texts: list[str], limit: int = CLIP_TOKEN_LIMIT, context: str = "") -> list[int]:
    """Raise if any rendering would be truncated by the CLIP text encoder.

    Returns the token length of every text so callers can log the distribution.
    """
    tok = _get_tokenizer()
    lengths = [len(ids) for ids in tok(list(texts))["input_ids"]]
    over = [(i, n) for i, n in enumerate(lengths) if n > limit]
    if over:
        i, n = over[0]
        raise ValueError(
            f"{len(over)} text(s) exceed the CLIP token limit ({limit}) {context}; "
            f"first offender index={i} tokens={n}: {texts[i][:120]!r}"
        )
    return lengths
