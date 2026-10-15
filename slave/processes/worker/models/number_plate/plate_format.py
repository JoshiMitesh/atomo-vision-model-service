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
# code) + 1-3 letters (series) + 4 digits (number), e.g. "MH12AB1234".
# A plate with no series letters is not a modern plate, so a read like
# "GJ320710" is a misread, not a real registration.
PLATE_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{4})$")

# Bharat (BH) series, e.g. "23BH1234A".
_BH_PLATE_RE = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")

_MIN_LENGTH = 7
_MAX_LENGTH = 12

STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL",
    "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UA", "UP", "WB",
}

# Highest RTO district code in use per state -- a read whose district is
# above it cannot be a real plate (e.g. "GJ93..." from a blurry GJ03).
# Only states listed here are range-checked; add others once confirmed.
# District 00 is never valid for any state.
MAX_DISTRICT = {"GJ": 39}

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


# Letters that can only ever be a misread digit: no Indian series uses I
# or O, so these are safe to turn back into digits wherever a digit is
# required. The wider TO_DIGIT map (G->6, S->5, B->8, A->4, T->7, ...) is
# NOT safe in the 4-digit tail or the district: G/S/B/A/T/Z/D are all
# legal series letters, so a plate whose last digit was clipped by the
# detector box ("GJ32AG288" for GJ32AG2883) used to have its series
# letter "repaired" into a digit and come out as a different, valid-
# looking plate ("GJ32A6288"). A read whose tail is not four real
# digits is now rejected instead of repaired.
SAFE_TO_DIGIT = {"O": "0", "I": "1"}

# A digit seen where a series letter belongs is a misread letter. Series
# letters never include I or O, so "0" can only be D (not O) and "1" has no
# valid letter at all.
SERIES_FROM_DIGIT = {"0": "D", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}


def normalize_plate(text: str) -> Tuple[str, bool]:
    """Force Indian layout: AA 00 A(AA) 0000. Returns (corrected_plate, valid).

    State letters are repaired (digit->letter look-alikes, then fix_state());
    the district and the 4-digit tail only accept I/O look-alikes. Anything
    else that is not already a digit there (a clipped or garbled read) is
    returned as invalid rather than guessed into a plate.
    """
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(t) < 8 or len(t) > _MAX_LENGTH:
        return t, False
    head = "".join(TO_LETTER.get(c, c) if c.isdigit() else c for c in t[:2])
    tail = "".join(SAFE_TO_DIGIT.get(c, c) for c in t[-4:])
    mid = t[2:-4]
    if not tail.isdigit():
        return fix_state(head) + mid + tail, False

    # district: 1-2 leading digits of the middle section (I/O look-alikes only)
    dist = ""
    for c in mid[:2]:
        c = SAFE_TO_DIGIT.get(c, c)
        if not c.isdigit():
            break
        dist += c
    # A 1-digit district followed by a digit look-alike is almost always a
    # misread 2-digit district ("GJ3ZK9938" for GJ32K9938) -- states like GJ
    # always use 2-digit codes. Only done when a series letter is still left.
    if len(dist) == 1 and len(mid) >= 3 and mid[1] in TO_DIGIT:
        dist += TO_DIGIT[mid[1]]
    # series: whatever is left must be letters (a stray digit is a misread letter)
    ser = "".join(SERIES_FROM_DIGIT.get(c, c) for c in mid[len(dist):])
    plate = fix_state(head) + dist + ser + tail
    valid = bool(PLATE_RE.match(plate)) and plate[:2] in STATES and not set(ser) & {"I", "O"}
    return plate, valid and _district_ok(plate[:2], dist)


def _district_ok(state: str, dist: str) -> bool:
    return 1 <= int(dist) <= MAX_DISTRICT.get(state, 99)


def is_valid_indian_plate(text: str) -> bool:
    """Plain format check with no correction -- kept for BH-series plates,
    which normalize_plate()'s district/series layout does not model."""
    if not text or len(text) < _MIN_LENGTH or len(text) > _MAX_LENGTH:
        return False
    return bool(PLATE_RE.match(text) or _BH_PLATE_RE.match(text))
