"""Global Yahoo Finance instrument search and input-normalization helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


CALENDAR_DAYS_PER_YEAR = 365.2425
TRADING_DAYS_PER_YEAR = 252


def parse_localized_number(value: Any) -> float:
    """Parse a number that may use either a decimal comma or decimal point."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "").replace("\u00a0", "").replace(" ", "")
    if not text:
        return float("nan")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    else:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parsed_weights(table: pd.DataFrame) -> pd.Series:
    """Return portfolio weights parsed from localized user input."""
    if "Weight (%)" not in table:
        return pd.Series(dtype=float)
    return table["Weight (%)"].map(parse_localized_number).astype(float)


def portfolio_weight_total(table: pd.DataFrame) -> float:
    """Return the valid entered portfolio-weight total."""
    weights = parsed_weights(table)
    return float(weights.dropna().sum()) if len(weights) else 0.0


def yahoo_asset_search(query: str, max_results: int = 12) -> list[dict[str, str]]:
    """Search all Yahoo Finance quote types by name or symbol."""
    normalized = query.strip()
    if len(normalized) < 2:
        return []

    import yfinance as yf

    translation = str.maketrans(
        {"ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G", "ç": "c", "Ç": "C", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U"}
    )

    def search_text(value: str) -> str:
        folded = unicodedata.normalize("NFKD", value.translate(translation))
        return re.sub(r"[^a-z0-9]+", " ", folded.encode("ascii", "ignore").decode().lower()).strip()

    ascii_query = search_text(normalized)
    variants = [normalized]
    if ascii_query and ascii_query != normalized.lower():
        variants.append(ascii_query)
    tokens = ascii_query.split()
    if len(tokens) > 2 and tokens[0] in {"turkiye", "turkey", "japan", "japanese"}:
        variants.append(" ".join(tokens[1:]))

    quotes: list[dict[str, Any]] = []
    for variant in dict.fromkeys(variants):
        search = yf.Search(
            variant,
            max_results=max_results,
            news_count=0,
            lists_count=0,
            include_cb=False,
            include_nav_links=False,
            include_research=False,
            enable_fuzzy_query=True,
            recommended=max_results,
            timeout=12,
            raise_errors=True,
        )
        quotes.extend(search.quotes)

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for quote in quotes:
        symbol = str(quote.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        name = str(quote.get("longname") or quote.get("shortname") or symbol).strip()
        raw_asset_type = str(
            quote.get("typeDisp") or quote.get("quoteType") or "Instrument"
        ).strip()
        type_labels = {
            "ETF": "ETF",
            "MUTUALFUND": "Mutual Fund",
            "FUTURE": "Futures",
            "INDEX": "Index",
            "EQUITY": "Equity",
            "CRYPTOCURRENCY": "Cryptocurrency",
            "CURRENCY": "Currency",
        }
        asset_type = type_labels.get(raw_asset_type.upper(), raw_asset_type.title())
        exchange = str(quote.get("exchDisp") or quote.get("exchange") or "Yahoo Finance").strip()
        rows.append(
            {
                "symbol": symbol,
                "name": name,
                "asset_type": asset_type.title(),
                "exchange": exchange,
            }
        )

    query_tokens = set(ascii_query.split())

    row_order = {row["symbol"]: index for index, row in enumerate(rows)}

    def relevance(row: dict[str, str]) -> float:
        symbol = search_text(row["symbol"])
        name = search_text(row["name"])
        combined_tokens = set(f"{symbol} {name}".split())
        overlap = len(query_tokens & combined_tokens) / max(len(query_tokens), 1)
        exact_symbol = float(symbol == ascii_query)
        phrase_match = float(bool(ascii_query) and ascii_query in name)
        starts_with = float(bool(ascii_query) and name.startswith(ascii_query))
        return exact_symbol * 100 + phrase_match * 50 + starts_with * 10 + overlap * 25

    rows.sort(key=lambda row: (-relevance(row), row_order[row["symbol"]]))
    return rows[:max_results]


def asset_option_label(asset: dict[str, str]) -> str:
    """Return a compact, globally informative search-result label."""
    return (
        f"{asset['name']} · {asset['symbol']} "
        f"· {asset['asset_type']} · {asset['exchange']}"
    )


def custom_horizon(amount: int, unit: str) -> tuple[int, str, float]:
    """Translate a calendar horizon into daily-resolution trading sessions."""
    amount = int(amount)
    if amount <= 0:
        raise ValueError("The simulation horizon must be greater than zero.")

    if unit == "Hours":
        calendar_days = amount / 24
    elif unit == "Days":
        calendar_days = amount
    elif unit == "Weeks":
        calendar_days = amount * 7
    elif unit == "Months":
        calendar_days = amount * CALENDAR_DAYS_PER_YEAR / 12
    elif unit == "Years":
        calendar_days = amount * CALENDAR_DAYS_PER_YEAR
    else:
        raise ValueError(f"Unsupported horizon unit: {unit}")

    trading_sessions = max(
        1,
        round(calendar_days * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR),
    )
    unit_label = unit.lower()
    if amount == 1:
        unit_label = {
            "hours": "hour",
            "days": "day",
            "weeks": "week",
            "months": "month",
            "years": "year",
        }[unit_label]
    label = f"{amount:,} {unit_label}"
    return int(trading_sessions), label, float(calendar_days)
