"""Attribute domains Omega_j and constraint set C (roadmap 2.4, F9).

Loads configs/schema.yaml and exposes:
  - normalize(field, value): canonicalise via aliases, lowercase enums
  - in_domain(field, value): value is a member of Omega_j
  - validate_attrs(category, attrs): full A' |= C check -> list of violations

Used by the Solver (refuse out-of-domain patches, roadmap 1.5), the VLM module
(validate new_value) and Verify (Eq. 27 schema-validity gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Violation:
    rule: str
    field: str
    value: Any
    message: str

    def to_dict(self) -> dict:
        return {"rule": self.rule, "field": self.field, "value": self.value, "message": self.message}


@dataclass
class Schema:
    attributes: dict = field(default_factory=dict)
    categories: list = field(default_factory=list)
    constraints: list = field(default_factory=list)

    # ---------- domain access ----------

    def domain(self, fld: str) -> list[str]:
        """Omega_j for an enum field; empty list for free-text/unknown fields."""
        spec = self.attributes.get(fld, {})
        if spec.get("type") == "enum":
            return [str(v) for v in spec.get("values", [])]
        return []

    def aliases(self, fld: str) -> dict[str, str]:
        spec = self.attributes.get(fld, {})
        return dict(spec.get("aliases", {}) or {}) if spec.get("type") == "enum" else {}

    def surface_forms(self, fld: str) -> list[str]:
        """Every string that may appear in text and normalise into Omega_j.

        Domain members plus alias keys, longest first so a multi-word form wins
        over a prefix of itself ("multi-color" before "multi"). Text scanners
        must use this rather than domain(): a title reading "Navy Shirt" carries
        a colour word even though `navy` is an alias, not a domain member.
        """
        forms = set(self.domain(fld)) | set(self.aliases(fld))
        return sorted(forms, key=lambda v: (-len(v), v))

    def checkable_fields(self) -> list[str]:
        """Enum fields, i.e. those with a finite domain a probe can enumerate."""
        return [k for k, v in self.attributes.items() if v.get("type") == "enum"]

    def normalize(self, fld: str, value: Any) -> str:
        spec = self.attributes.get(fld, {})
        s = str(value).strip()
        if spec.get("type") == "enum":
            s = s.lower()
            s = str(spec.get("aliases", {}).get(s, s))
        return s

    def in_domain(self, fld: str, value: Any) -> bool:
        spec = self.attributes.get(fld, None)
        if spec is None:
            return False
        if spec.get("type") == "free_text":
            return len(str(value)) <= int(spec.get("max_len", 10_000))
        domain_norm = {self.normalize(fld, v) for v in self.domain(fld)}
        return self.normalize(fld, value) in domain_norm

    # ---------- constraint set C ----------

    def validate_attrs(self, category: str, attrs: dict) -> list[Violation]:
        """Check A' |= C. Returns [] when valid."""
        out: list[Violation] = []

        if self.categories and category not in self.categories:
            out.append(Violation("known_category", "category", category, f"unknown category: {category!r}"))

        for fld, spec in self.attributes.items():
            # Simple required: applies to all categories
            if spec.get("required") and (fld not in attrs or attrs.get(fld) in (None, "", [])):
                out.append(Violation("required", fld, None, f"required attribute {fld!r} is missing"))
            # Category-scoped required: applies only when category is in the list
            req_cats = spec.get("required_for_categories", [])
            if req_cats and category in req_cats:
                if fld not in attrs or attrs.get(fld) in (None, "", []):
                    out.append(Violation("required", fld, None,
                                        f"required attribute {fld!r} is missing for category {category!r}"))

        for fld, val in attrs.items():
            if fld not in self.attributes:
                continue  # unknown attributes are tolerated, not validated
            if not self.in_domain(fld, val):
                out.append(
                    Violation("domain", fld, val, f"{fld}={val!r} not in Omega_{fld} = {self.domain(fld)}")
                )

        for rule in self.constraints:
            cond = rule.get("if", {})
            if not all(str(attrs.get(k, category if k == "category" else None)) == str(v) for k, v in cond.items()):
                continue
            for fld, allowed in rule.get("allow", {}).items():
                if fld in attrs and self.normalize(fld, attrs[fld]) not in {self.normalize(fld, a) for a in allowed}:
                    out.append(
                        Violation(rule.get("name", "constraint"), fld, attrs[fld],
                                  f"{fld}={attrs[fld]!r} violates {rule.get('name')} for {cond}")
                    )
        return out

    def is_valid(self, category: str, attrs: dict) -> bool:
        return not self.validate_attrs(category, attrs)


def _assert_domains_disjoint_from_aliases(attributes: dict) -> None:
    """Omega_j and the alias keys must not intersect (finding D5).

    A value that is both a domain member and an alias of another value is a
    semantic duplicate: probes enumerate it as a rival candidate for the same
    evidence, and every raw-vs-normalised comparison in the codebase silently
    disagrees with itself. Refuse to load rather than degrade quietly.
    """
    for fld, spec in (attributes or {}).items():
        if (spec or {}).get("type") != "enum":
            continue
        values = {str(v).strip().lower() for v in (spec.get("values") or [])}
        alias_keys = {str(k).strip().lower() for k in (spec.get("aliases") or {})}
        clash = sorted(values & alias_keys)
        if clash:
            raise ValueError(
                f"schema attribute {fld!r}: {clash} appear in both `values` and "
                f"`aliases`. A canonical value cannot also be an alias of another "
                f"value. Keep each surface form in exactly one place."
            )


def load_schema(path: str | Path) -> Schema:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    attributes = obj.get("attributes", {}) or {}
    _assert_domains_disjoint_from_aliases(attributes)
    return Schema(
        attributes=attributes,
        categories=[str(c) for c in (obj.get("categories", []) or [])],
        constraints=obj.get("constraints", []) or [],
    )
