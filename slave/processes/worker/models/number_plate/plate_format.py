"""Indian vehicle registration plate format validation + correction.

Extended from the original is_valid_indian_plate() (ported from the
reference ALPR script's is_valid_indian_plate(), alpr_rtsp.py) with
anpr_night's look-alike correction (anpr_night/anpr.py's fix_state()/
normalize()) -- live night-CCTV testing showed OCR mistakes are
concentrated in a few confusable character pairs (0/O, 1/I, 5/S, ...)
at predictable positions (state letters vs digits), so correcting those
before validating recovers plates that a plain regex check would
otherwise just reject.

Kept in its own module (not plate_ocr.py) since it is a country-specific
business rule, not part of the OCR pixel pipeline -- a non-Indian
deployment swaps this module (or flips config/models/number_plate.yaml's
`ocr.validate_indian_plate` off) without touching plate_ocr.py.
"""

from __future__ import annotations

import re
from typing import Tuple

# Normal state-issued registration: 2 letters (state) + 1-2 digits (RTO
# code) + 0-3 letters (series) + 4 digits (number), e.g. "MH12AB1234".
PLATE_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})$")

# Bharat (BH) series, e.g. "23BH1234A".
_BH_PLATE_RE = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")

_MIN_LENGTH = 7
_MAX_LENGTH = 12

STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL",
    "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UA", "UP", "WB",
}

TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A", "7": "T"}
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "T": "7", "A": "4"}
LOOKALIKE = {"G": "CO06", "C": "G0O", "O": "0DQ", "D": "0O", "B": "8", "S": "5", "Z": "2", "I": "1L"}


def fix_state(s: str) -> str:
    """Repair the 2-letter state code using look-alike characters (0J/CJ -> GJ)."""
    if s in STATES:
        return s
    for i in (0, 1):
        for real, fakes in LOOKALIKE.items():
            if s[i] in fakes:
                t = s[:i] + real + s[i + 1:]
                if t in STATES:
                    return t
    return s


def normalize_plate(text: str) -> Tuple[str, bool]:
    """Force Indian layout: AA 00 A(AA) 0000. Returns (corrected_plate, valid)."""
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(t) < 8:
        return t, False
    head = "".join(TO_LETTER.get(c, c) if c.isdigit() else c for c in t[:2])
    tail = "".join(TO_DIGIT.get(c, c) for c in t[-4:])
    mid = t[2:-4]
    # district: first 1-2 chars as digits, series: rest as letters
    m = re.match(
        r"^(.{1,2}?)(\D*)$", "".join(TO_DIGIT.get(c, c) if i < 2 else c for i, c in enumerate(mid))
    ) if mid else None
    dist, ser = (m.group(1), m.group(2)) if m else (mid, "")
    ser = "".join(TO_LETTER.get(c, c) for c in ser)
    plate = fix_state(head) + dist + ser + tail
    return plate, bool(PLATE_RE.match(plate)) and plate[:2] in STATES


def is_valid_indian_plate(text: str) -> bool:
    """Plain format check with no correction -- kept for BH-series plates,
    which normalize_plate()'s district/series layout does not model."""
    if not text or len(text) < _MIN_LENGTH or len(text) > _MAX_LENGTH:
        return False
    return bool(PLATE_RE.match(text) or _BH_PLATE_RE.match(text))
