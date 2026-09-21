"""Indian vehicle registration plate format validation.

Ported from the updated reference ALPR script's is_valid_indian_plate()
(alpr_rtsp.py) -- live RTSP testing showed PaddleOCR occasionally
returns plausible-looking but wrong text (right length, wrong format);
gating on the actual registration format catches those before they
ever reach a track's vote history.

Kept in its own module (not plate_ocr.py) since it is a country-specific
business rule, not part of the OCR pixel pipeline -- a non-Indian
deployment swaps this module (or flips config/models/number_plate.yaml's
`ocr.validate_indian_plate` off) without touching plate_ocr.py.
"""

from __future__ import annotations

import re

# Normal state-issued registration: 2 letters (state) + 1-2 digits (RTO
# code) + 1-3 letters (series) + 1-4 digits (number), e.g. "MH12AB1234".
_NORMAL_PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$")

# Bharat (BH) series, e.g. "23BH1234A".
_BH_PLATE_RE = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")

_MIN_LENGTH = 7
_MAX_LENGTH = 12


def is_valid_indian_plate(text: str) -> bool:
    if not text or len(text) < _MIN_LENGTH or len(text) > _MAX_LENGTH:
        return False
    return bool(_NORMAL_PLATE_RE.match(text) or _BH_PLATE_RE.match(text))
