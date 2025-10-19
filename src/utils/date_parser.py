"""Utilidades para detectar y normalizar fechas en textos OCR."""

from __future__ import annotations

import re
from typing import Optional

_MONTHS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}

_NUMERIC_PATTERNS = [
    re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"),
    re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b"),
]

_TEXTUAL_PATTERN = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{2,4})\b", re.IGNORECASE)


def _normalize_year(year: str) -> str:
    year = year.strip()
    if len(year) == 2:
        return f"20{year}"
    return year.zfill(4)


def _normalize_unit(value: str) -> str:
    value = value.strip()
    return value.zfill(2)


def _build_date(day: str, month: str, year: str) -> Optional[str]:
    try:
        day_norm = _normalize_unit(day)
        month_norm = _normalize_unit(month)
        year_norm = _normalize_year(year)
        return f"{day_norm}/{month_norm}/{year_norm}"
    except ValueError:
        return None


def find_date_in_text(text: str) -> Optional[str]:
    """
    Busca la primera fecha reconocible en el texto y la normaliza a DD/MM/YYYY.
    """
    if not text:
        return None

    for pattern in _NUMERIC_PATTERNS:
        match = pattern.search(text)
        if match:
            day, month, year = match.groups()
            return _build_date(day, month, year)

    textual_match = _TEXTUAL_PATTERN.search(text)
    if textual_match:
        day, month_name, year = textual_match.groups()
        month_key = month_name.lower()
        month_norm = _MONTHS.get(month_key[:3]) or _MONTHS.get(month_key)
        if month_norm:
            return _build_date(day, month_norm, year)

    return None
