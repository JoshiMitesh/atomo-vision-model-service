"""Plate-text cleanup for the NPU (`.nb`/`.so`) number_plate path.

Ported from the board-verified alpr_npu.py (normalize_ocr,
is_indian_plate, generate_corrections, choose_best_plate_candidate).
Deliberately separate from plate_format.py, which the `.pt` path uses
with different (stricter, length-bounded) rules -- the two pipelines
are kept independent so tuning one never changes the other.
"""

from __future__ import annotations

import re
from typing import List, Tuple

NORMAL_PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$")
BH_PLATE_RE = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")

# Letter -> the digit OCR commonly confuses it with.
OCR_CONFUSIONS = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "7",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}


def normalize_ocr(text) -> str:
    if text is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def is_indian_plate(text: str) -> bool:
    if not text:
        return False
    return bool(NORMAL_PLATE_RE.fullmatch(text) or BH_PLATE_RE.fullmatch(text))


def generate_corrections(text: str) -> List[str]:
    """A small number of conservative OCR alternatives. Letters and
    digits both legitimately occur in Indian plates, so this does NOT
    blindly replace every O with 0 -- it only tries the usual confusions
    around the positions where the registration format expects a digit.
    """
    text = normalize_ocr(text)
    if not text:
        return []

    candidates = [text]

    # Normal registration: XX 00 XX 0000
    if len(text) >= 8:
        chars = list(text)

        # First two characters should be state letters.
        for i in range(min(2, len(chars))):
            if chars[i] == "0":
                chars[i] = "O"
            elif chars[i] == "1":
                chars[i] = "I"
        candidates.append("".join(chars))

        # Numeric positions right after the state code.
        for digit_end in (4, 5):
            if len(text) >= digit_end:
                chars = list(text)
                for i in range(2, min(digit_end, len(chars))):
                    if chars[i] in OCR_CONFUSIONS:
                        chars[i] = OCR_CONFUSIONS[chars[i]]
                candidates.append("".join(chars))

    # Generic single-character alternatives.
    for i, ch in enumerate(text):
        if ch in OCR_CONFUSIONS:
            alt = list(text)
            alt[i] = OCR_CONFUSIONS[ch]
            candidates.append("".join(alt))

    unique: List[str] = []
    for c in candidates:
        if c not in unique:
            unique.append(c)
    return unique[:12]


def choose_best_plate_candidate(raw_text, raw_conf: float) -> Tuple[str, float, bool]:
    """Priority: exact Indian format, then corrected Indian format
    (confidence x0.88), otherwise the normalized text as-is (invalid).
    Returns (text, confidence, valid).
    """
    raw = normalize_ocr(raw_text)
    if not raw:
        return "", 0.0, False

    if is_indian_plate(raw):
        return raw, float(raw_conf), True

    valid_candidates = []
    for candidate in generate_corrections(raw):
        if is_indian_plate(candidate):
            penalty = 0.96 if candidate == raw else 0.88
            valid_candidates.append((float(raw_conf) * penalty, candidate))

    if valid_candidates:
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        score, candidate = valid_candidates[0]
        return candidate, score, True

    return raw, float(raw_conf), False
