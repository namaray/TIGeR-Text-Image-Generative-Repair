"""Normalise ABO's free-text colour and material strings into Omega_j.

ABO attribute values are written by sellers, not drawn from a controlled
vocabulary: the two furnishing verticals carry 2,147 distinct colour strings and
608 material strings over ~13.5k products, against a 12-value colour domain and
a 17-value material domain. Importing them raw makes every value out-of-domain,
so `schema.validate_attrs` rejects the row and it escalates before any repair is
attempted -- the same failure mode as H10, one layer earlier.

Three kinds of noise, handled in order:

  1. modifiers      "light grey", "matte black", "dark bronze"  -> strip, keep head
  2. other languages "blanco", "negro", "gris", "azul", "metall" -> alias
  3. non-values     "no aplica", "not applicable", "other"       -> None, not a guess

Returning None is deliberate. An unresolvable value must drop the attribute so
the row is repaired or escalated on its merits; mapping it to a plausible-looking
colour would fabricate ground truth and silently inflate restoration accuracy.
"""

from __future__ import annotations

import re

# Values that explicitly mean "not stated". Never guess these.
_NULL = {
    "no aplica", "not applicable", "n/a", "na", "none", "other", "不适用",
    "fine-other-material", "no aplicable", "sin especificar", "unknown",
    "assorted", "as shown", "as pictured", "see description",
    "not-applicable", "non applicable", "non applicable.", "non applicabile",
    "nicht zutreffend", "sonstiges", "altro", "autre", "その他", "無し",
    "no aplica.", "varios", "various", "misc", "miscellaneous", "n.a.",
}

# Leading qualifiers that do not change the base colour.
_MODIFIERS = {
    "light", "dark", "deep", "pale", "bright", "matte", "glossy", "satin",
    "brushed", "polished", "antique", "vintage", "distressed", "warm", "cool",
    "soft", "rich", "solid", "classic", "true", "medium", "heather", "faux",
    "textured", "weathered", "rustic", "aged", "burnished", "oil-rubbed",
    "oil", "rubbed", "plated", "plate", "tone", "toned", "finish", "colored",
    "coloured", "color", "colour",
}

_COLOR_ALIASES = {
    # spelling / regional
    "grey": "gray", "gray": "gray", "multicolour": "multicolour",
    "multicolor": "multicolour", "multi": "multicolour", "multi-color": "multicolour",
    "multi color": "multicolour", "assorted colors": "multicolour",
    # other languages
    "blanco": "white", "negro": "black", "gris": "gray", "azul": "blue",
    "rojo": "red", "verde": "green", "amarillo": "yellow", "marron": "brown",
    "rosa": "pink", "morado": "purple", "naranja": "orange", "plata": "gray",
    "dorado": "yellow", "bianco": "white", "nero": "black", "grigio": "gray",
    "blau": "blue", "schwarz": "black", "weiss": "white", "weiß": "white",
    "noir": "black", "blanc": "white", "bleu": "blue", "vert": "green",
    # neutrals and naturals
    "beige": "white", "cream": "white", "ivory": "white", "off-white": "white",
    "eggshell": "white", "linen": "white", "natural": "brown", "sand": "brown",
    "tan": "brown", "taupe": "brown", "camel": "brown", "cognac": "brown",
    "chestnut": "brown", "walnut": "brown", "oak": "brown", "espresso": "brown",
    "mocha": "brown", "khaki": "brown", "wheat": "brown", "honey": "brown",
    "charcoal": "gray", "slate": "gray", "stone": "gray", "graphite": "gray",
    "pewter": "gray", "ash": "gray", "smoke": "gray", "greige": "gray",
    # metals -> nearest domain colour
    "silver": "gray", "chrome": "gray", "nickel": "gray", "steel": "gray",
    "stainless": "gray", "stainless steel": "gray", "platinum": "gray",
    "gunmetal": "gray", "aluminum": "gray", "titanium": "gray",
    "gold": "yellow", "brass": "yellow", "bronze": "brown", "copper": "brown",
    "rose gold": "pink", "rose": "pink", "blush": "pink", "coral": "pink",
    "salmon": "pink",
    # chromatics
    "navy": "blue", "teal": "green", "turquoise": "blue", "aqua": "blue",
    "cyan": "blue", "indigo": "blue", "cobalt": "blue", "sky": "blue",
    "denim": "blue", "royal": "blue", "sapphire": "blue",
    "burgundy": "red", "maroon": "red", "wine": "red", "crimson": "red",
    "scarlet": "red", "ruby": "red", "rust": "orange", "terracotta": "orange",
    "amber": "orange", "apricot": "orange", "peach": "orange",
    "olive": "green", "sage": "green", "mint": "green", "emerald": "green",
    "forest": "green", "lime": "green", "jade": "green",
    "lavender": "purple", "lilac": "purple", "violet": "purple",
    "plum": "purple", "magenta": "purple", "mauve": "purple",
    "mustard": "yellow", "lemon": "yellow", "champagne": "yellow",
    "clear": "white", "transparent": "white", "glass": "white",
    # further languages seen in the ABO furnishing verticals
    "ホワイト": "white", "ブラック": "black", "シルバー": "gray",
    "ゴールド": "yellow", "ブラウン": "brown", "ブルー": "blue",
    "silber": "gray", "weis": "white", "braun": "brown", "grau": "gray",
    "gelb": "yellow", "rot": "red", "gruen": "green", "grün": "green",
    "plateado": "gray", "dourado": "yellow", "prata": "gray", "preto": "black",
    "branco": "white", "cinza": "gray", "castanho": "brown",
    # naturals and neutrals in the tail
    "hemp": "brown", "saddle": "brown", "driftwood": "brown", "chalk": "white",
    "ecru": "white", "dove": "gray", "pearl": "white", "bone": "white",
    "alabaster": "white", "snow": "white", "frost": "white", "vanilla": "white",
    "biscuit": "brown", "caramel": "brown", "toffee": "brown", "cedar": "brown",
    "mahogany": "brown", "hazel": "brown", "sepia": "brown", "umber": "brown",
    "amethyst": "purple", "aubergine": "purple", "eggplant": "purple",
    "onyx": "black", "ebony": "black", "jet": "black", "ink": "black",
    "midnight": "black", "obsidian": "black", "raven": "black",
    "cerulean": "blue", "periwinkle": "blue", "denim blue": "blue",
    "seafoam": "green", "moss": "green", "fern": "green", "hunter": "green",
    "brick": "red", "cherry": "red", "cranberry": "red", "garnet": "red",
    "tangerine": "orange", "pumpkin": "orange", "ginger": "orange",
}

_MATERIAL_ALIASES = {
    "metall": "metal", "metallo": "metal", "metal": "metal", "aluminum": "metal",
    "aluminium": "metal", "steel": "metal", "stainless steel": "metal",
    "iron": "metal", "brass": "metal", "bronze": "metal", "copper": "metal",
    "alloy": "metal", "zinc": "metal", "sterling silver": "metal",
    "silver": "metal", "gold": "metal", "platinum": "metal", "nickel": "metal",
    "madera": "wood", "holz": "wood", "legno": "wood", "engineered wood": "wood",
    "solid wood": "wood", "pine": "wood", "oak": "wood", "walnut": "wood",
    "birch": "wood", "teak": "wood", "acacia": "wood", "mango wood": "wood",
    "rubberwood": "wood", "plywood": "wood", "mdf": "wood", "bamboo": "wood",
    "faux leather": "leather", "pu leather": "leather", "bonded leather": "leather",
    "genuine leather": "leather", "cuero": "leather", "leder": "leather",
    "tela": "fabric", "textile": "fabric", "upholstery": "fabric",
    "velvet": "fabric", "linen": "fabric", "jute": "fabric", "chenille": "fabric",
    "microfiber": "fabric", "faux fur": "fabric", "felt": "fabric",
    "polypropylene": "plastic", "acrylic": "plastic", "resin": "plastic",
    "pvc": "plastic", "vinyl": "plastic", "melamine": "plastic",
    "porcelain": "ceramic", "stoneware": "ceramic", "earthenware": "ceramic",
    "marble": "stone", "granite": "stone", "concrete": "stone", "slate": "stone",
    "cardboard": "paper", "paperboard": "paper",
    "cristal": "glass", "tempered glass": "glass",
    "abs": "plastic", "abs plastic": "plastic", "polycarbonate": "plastic",
    "polyethylene": "plastic", "silicone": "silicone", "rubber": "rubber",
    "nylon": "fabric", "suede": "suede", "faux suede": "suede",
    "microsuede": "fabric", "cashmere": "wool", "sherpa": "wool",
    "denim": "denim", "twill": "fabric", "canvas": "canvas",
    "rattan": "wood", "wicker": "wood", "seagrass": "fabric",
    "marmol": "stone", "granit": "stone", "quartz": "stone",
    "sandstone": "stone", "limestone": "stone", "travertine": "stone",
    "cemento": "stone", "terrazzo": "stone",
}

_SPLIT = re.compile(r"[\s/,&|+()-]+")


def _canon(raw: str, aliases: dict[str, str], domain: set[str]) -> str | None:
    """Longest-match-first canonicalisation of a free-text attribute value."""
    if raw is None:
        return None
    s = " ".join(str(raw).strip().lower().split())
    if not s or s in _NULL:
        return None

    if s in aliases:
        return aliases[s]
    if s in domain:
        return s

    tokens = [t for t in _SPLIT.split(s) if t]
    if not tokens:
        return None

    # longest contiguous span that resolves, preferring specific over general
    for width in range(min(3, len(tokens)), 0, -1):
        for i in range(len(tokens) - width + 1):
            span = " ".join(tokens[i:i + width])
            if span in aliases:
                return aliases[span]
            if span in domain:
                return span

    # last resort: drop modifiers, retry on the remaining head tokens
    head = [t for t in tokens if t not in _MODIFIERS]
    for t in head:
        if t in aliases:
            return aliases[t]
        if t in domain:
            return t
    return None


def normalize_color(raw, domain: set[str] | None = None) -> str | None:
    return _canon(raw, _COLOR_ALIASES, domain or set())


def normalize_material(raw, domain: set[str] | None = None) -> str | None:
    return _canon(raw, _MATERIAL_ALIASES, domain or set())
