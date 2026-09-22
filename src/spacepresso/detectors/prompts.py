"""Text-prompt construction for the language-grounded detectors.

WinCLIP and TextAD both need a set of "this object is normal" and "this object
is defective" sentences per class, and both read the project's
``data/anomaly_descriptions.csv`` to ground those sentences in the defect
types that actually occur. The construction lived inside
``winclip_baseline.py``, which is why TextAD carried its own near-copy.

The prompt ensemble follows WinCLIP's Compositional Prompt Ensemble: a grid of
state words crossed with photo templates, averaged in embedding space. Single
prompts are noisy — CLIP's text encoder is sensitive to phrasing in ways that
have nothing to do with the semantics — and averaging dozens of paraphrases
cancels most of that out.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ClassDescriptions",
    "build_prompts",
    "extract_keywords",
    "load_descriptions",
]

TEMPLATES: tuple[str, ...] = (
    "a photo of a {state} {obj}",
    "a photo of the {state} {obj}",
    "a close-up photo of a {state} {obj}",
    "a cropped photo of a {state} {obj}",
    "a bright photo of a {state} {obj}",
    "a dark photo of a {state} {obj}",
    "a blurry photo of a {state} {obj}",
    "a photo of a small {state} {obj}",
    "a photo of a large {state} {obj}",
    "an industrial inspection photo of a {state} {obj}",
)

NORMAL_STATES: tuple[str, ...] = (
    "normal",
    "perfect",
    "flawless",
    "unblemished",
    "undamaged",
    "good",
)

ANOMALY_STATES: tuple[str, ...] = (
    "damaged",
    "defective",
    "broken",
    "flawed",
    "anomalous",
    "with a defect",
)

#: Substrings that reliably imply a defect, mapped to a short noun phrase.
#: Short phrases work better than whole descriptions: CLIP's training corpus
#: contains "a crack" far more often than a two-clause inspection report.
KEYWORD_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("scratch", "scrape", "abrasion"), "a scratch"),
    (("dent", "deform", "bent", "warp"), "a dent"),
    (("crack", "fissur"), "a crack"),
    (("mold", "fungal", "fuzzy", "powdery"), "mold"),
    (("infest", "pest", "insect"), "pest damage"),
    (("contamination", "soil", "dirt", "oil residue"), "contamination"),
    (("compress", "flatten"), "a flattened area"),
    (("layered cap", "additional cap", "extra cap"), "an extra cap layer"),
    (("hole", "pitted"), "holes"),
    (("rough edge", "jagged edge"), "rough edges"),
    (("rust",), "rust"),
    (("discolour", "discolor", "stain"), "discolouration"),
    (("chip", "chipped"), "a chipped edge"),
    (("missing", "absent"), "a missing part"),
)

#: Descriptions containing this are placeholders and add no signal.
GENERIC_PHRASE = "an anomaly is present"

#: Fallback object names when the CSV is unavailable.
CLASS_FALLBACK_NAME: dict[str, str] = {
    "class_01": "resistor",
    "class_02": "inductor",
    "class_03": "gear",
    "class_04": "screw",
    "class_05": "nut",
    "class_06": "coffee bean",
    "class_07": "pistachio",
    "class_08": "capsule",
}


@dataclass(frozen=True, slots=True)
class ClassDescriptions:
    object_name: str
    per_type: dict[str, str]


def extract_keywords(description: str) -> list[str]:
    """Short defect noun-phrases implied by a description.

    Conservative by design: only substrings with unambiguous defect semantics
    trigger, so a description that says "no scratches were found" cannot add a
    "a scratch" prompt through a looser match.
    """
    lowered = description.lower()
    found: list[str] = []
    seen: set[str] = set()
    for triggers, phrase in KEYWORD_RULES:
        if phrase in seen:
            continue
        if any(trigger in lowered for trigger in triggers):
            found.append(phrase)
            seen.add(phrase)
    return found


def build_prompts(
    object_name: str, per_type: dict[str, str] | None = None
) -> tuple[list[str], list[str]]:
    """``(normal_prompts, anomaly_prompts)`` for one class."""
    normal = [
        template.format(state=state, obj=object_name)
        for state in NORMAL_STATES
        for template in TEMPLATES
    ]
    anomaly = [
        template.format(state=state, obj=object_name)
        for state in ANOMALY_STATES
        for template in TEMPLATES
    ]

    seen_descriptions: set[str] = set()
    seen_keywords: set[str] = set()
    for description in (per_type or {}).values():
        if not description or GENERIC_PHRASE in description.lower():
            continue
        text = description.strip()
        if text not in seen_descriptions:
            anomaly.append(f"a photo of a defective {object_name}: {text}")
            anomaly.append(f"a photo of an anomalous {object_name}: {text}")
            seen_descriptions.add(text)
        for keyword in extract_keywords(text):
            if keyword in seen_keywords:
                continue
            seen_keywords.add(keyword)
            anomaly.append(f"a photo of a {object_name} with {keyword}")
            anomaly.append(f"a {object_name} showing {keyword}")
            anomaly.append(f"a close-up photo of {keyword} on a {object_name}")

    return normal, anomaly


def load_descriptions(csv_path: Path | None) -> dict[str, ClassDescriptions]:
    """Parse ``anomaly_descriptions.csv``.

    Returns an empty mapping when the file is missing, so callers fall back to
    pure template prompts rather than failing — the descriptions improve the
    prompts but are not required for the detector to run.
    """
    if csv_path is None or not Path(csv_path).is_file():
        return {}

    object_names: dict[str, str] = {}
    per_type: dict[str, dict[str, str]] = {}

    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cls = (row.get("class") or row.get("class_name") or "").strip()
            if not cls:
                continue
            name = (row.get("object") or row.get("object_name") or "").strip()
            if name:
                object_names.setdefault(cls, name)
            anomaly_type = (row.get("anomaly_type") or "").strip()
            description = (row.get("description") or "").strip()
            if anomaly_type and description:
                per_type.setdefault(cls, {})[anomaly_type] = description

    return {
        cls: ClassDescriptions(
            object_name=object_names.get(cls, CLASS_FALLBACK_NAME.get(cls, "object")),
            per_type=per_type.get(cls, {}),
        )
        for cls in set(object_names) | set(per_type)
    }
