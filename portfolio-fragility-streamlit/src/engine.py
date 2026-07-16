"""Portfolio simulation, regime risk, liquidity cascades, and stress testing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .analytics import calendar_time_cagr
from .instruments import parsed_weights


TRADING_DAYS = 252
CALENDAR_DAYS_PER_YEAR = 365.2425
REGIME_NAMES = ("Bull", "Normal", "Crisis", "Recovery")
BASE_CURRENCY = "USD"
_CURRENCY_SUBUNITS = {
    "GBP": ("GBP", 1.0),
    "GBPENCE": ("GBP", 0.01),
    "GBPENNY": ("GBP", 0.01),
    "GBX": ("GBP", 0.01),
    "ILA": ("ILS", 0.01),
    "ZAC": ("ZAR", 0.01),
}


class PortfolioError(ValueError):
    """User-facing validation or market-data error."""


@dataclass(frozen=True)
class AnalysisConfig:
    tickers: tuple[str, ...]
    weights: tuple[float, ...]
    benchmark: str
    start_date: date
    initial_investment: float
    simulation_days: int
    simulations: int
    confidence: float
    target_value: float
    risk_free_rate: float
    method: str
    seed: int = 42
    student_df: int = 5
    redemption_probability: float = 0.35
    redemption_pct: float = 0.12
    margin_call_pct: float = 0.05
    margin_trigger: float = -0.15
    cash_buffer_pct: float = 0.03
    gross_leverage: float = 1.0
    horizon_label: str = ""
    calendar_days: float = 0.0
    base_currency: str = BASE_CURRENCY


@dataclass
class AnalysisResult:
    prices: pd.DataFrame
    asset_returns: pd.DataFrame
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    paths: np.ndarray
    terminal_values: np.ndarray
    path_percentiles: pd.DataFrame
    metrics: dict[str, float]
    risk_contributions: pd.DataFrame
    liquidity: pd.DataFrame
    stress_tests: pd.DataFrame
    regime_summary: pd.DataFrame
    fragility_components: pd.DataFrame
    allocation_comparison: pd.DataFrame
    resilient_weights: np.ndarray


def validate_portfolio(table: pd.DataFrame) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Validate an editable ticker/weight table using percentage weights."""
    data = table.copy().dropna(how="all")
    data["Ticker"] = data["Ticker"].astype(str).str.strip().str.upper()
    data = data[data["Ticker"].ne("") & data["Ticker"].ne("NAN")]
    if len(data) < 2:
        raise PortfolioError("Add at least two different assets.")
    if data["Ticker"].duplicated().any():
        raise PortfolioError("The same ticker cannot be used more than once.")
    weights = parsed_weights(data)
    if (
        weights.isna().any()
        or not np.all(np.isfinite(weights))
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise PortfolioError(
            "Weights must be non-negative numbers. Both 20.3 and 20,3 are accepted."
        )
    if not np.isclose(weights.sum(), 100, atol=0.05):
        raise PortfolioError(f"Weights must total 100%. Current total: {weights.sum():.2f}%")
    return tuple(data["Ticker"]), tuple((weights / 100).to_numpy())


def _field_frame(raw: pd.DataFrame, field: str, symbols: list[str]) -> pd.DataFrame:
    """Return one yfinance field with normalized symbol columns."""
    if raw.empty:
        return pd.DataFrame(index=raw.index)

    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            frame = raw.xs(field, level=0, axis=1, drop_level=True)
        elif field in raw.columns.get_level_values(1):
            frame = raw.xs(field, level=1, axis=1, drop_level=True)
        else:
            return pd.DataFrame(index=raw.index)
    elif field in raw.columns:
        selected = raw[field]
        frame = (
            selected.to_frame(symbols[0])
            if isinstance(selected, pd.Series)
            else selected.copy()
        )
    else:
        return pd.DataFrame(index=raw.index)

    if isinstance(frame, pd.Series):
        frame = frame.to_frame(symbols[0])
    elif len(symbols) == 1 and len(frame.columns) == 1:
        frame = frame.rename(columns={frame.columns[0]: symbols[0]})
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    return frame


def _metadata_profile(yf: Any, symbol: str) -> dict[str, str] | None:
    """Read Yahoo currency and instrument type without relying on one endpoint."""
    ticker = yf.Ticker(symbol)
    currency: Any = None
    instrument_type: Any = None
    try:
        fast_info = ticker.fast_info
        currency = (
            fast_info.get("currency")
            if hasattr(fast_info, "get")
            else getattr(fast_info, "currency", None)
        )
    except Exception:
        currency = None
    try:
        metadata = ticker.get_history_metadata()
        if isinstance(metadata, dict):
            if not currency:
                currency = metadata.get("currency")
            instrument_type = metadata.get("instrumentType")
    except Exception:
        metadata = None
    if not currency:
        return None
    if symbol.strip().upper().endswith("=X"):
        instrument_type = "CURRENCY"
    return {
        "currency": str(currency).strip(),
        "instrument_type": str(instrument_type or "UNKNOWN").strip().upper(),
    }


def _symbol_metadata(yf: Any, symbols: list[str]) -> dict[str, dict[str, str]]:
    """Fetch Yahoo instrument metadata concurrently and fail before mixing currencies."""
    profiles: dict[str, dict[str, str]] = {}
    workers = min(8, max(1, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_metadata_profile, yf, symbol): symbol for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                profile = future.result()
            except Exception:
                profile = None
            if profile:
                profiles[symbol] = profile

    missing = [symbol for symbol in symbols if symbol not in profiles]
    if missing:
        raise PortfolioError(
            "Yahoo Finance did not provide currency metadata for: "
            + ", ".join(missing)
            + ". Currency metadata is required to convert every series to USD safely."
        )
    return profiles


def _normalize_currency(currency: str) -> tuple[str, float]:
    """Return the major currency and local-price unit factor."""
    normalized = currency.strip().upper().replace(" ", "")
    if normalized in {"GBP", "GBX"}:
        # Yahoo uses mixed casing to distinguish pounds (GBP) from pence (GBp).
        if currency.strip() in {"GBp", "GBX", "GBx"}:
            return "GBP", 0.01
        return "GBP", 1.0
    if normalized in _CURRENCY_SUBUNITS:
        return _CURRENCY_SUBUNITS[normalized]
    return normalized, 1.0


def _download_fx_rate(yf: Any, currency: str, start_date: date) -> pd.Series:
    """Download USD per unit of a currency, using an inverse Yahoo pair as fallback."""
    attempts = (
        (f"{currency}{BASE_CURRENCY}=X", False),
        (f"{BASE_CURRENCY}{currency}=X", True),
    )
    for pair, inverse in attempts:
        try:
            raw = yf.download(
                pair,
                start=start_date.isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
            )
        except Exception:
            continue
        close = _field_frame(raw, "Close", [pair])
        if pair not in close:
            continue
        rate = pd.to_numeric(close[pair], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        rate = rate.where(rate > 0).dropna()
        if rate.empty:
            continue
        rate.index = pd.to_datetime(rate.index).tz_localize(None)
        rate = rate.sort_index()
        return (1.0 / rate if inverse else rate).rename(currency)
    raise PortfolioError(
        f"Yahoo Finance could not provide a {currency}/USD exchange-rate history. "
        f"The affected assets cannot be combined safely in a USD portfolio."
    )


def _convert_prices_to_usd(
    close: pd.DataFrame,
    metadata: dict[str, dict[str, str]],
    yf: Any,
    start_date: date,
) -> pd.DataFrame:
    """Convert Yahoo price histories to USD while leaving share volumes unchanged."""
    conversion_specs = {
        symbol: _normalize_currency(metadata[symbol]["currency"]) for symbol in close.columns
    }
    needed_currencies = sorted(
        {
            major
            for symbol, (major, _) in conversion_specs.items()
            if major != BASE_CURRENCY
            and metadata[symbol]["instrument_type"] != "CURRENCY"
        }
    )
    fx_rates = {
        currency: _download_fx_rate(yf, currency, start_date)
        for currency in needed_currencies
    }

    converted = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for symbol in close.columns:
        major, unit_factor = conversion_specs[symbol]
        local_price = pd.to_numeric(close[symbol], errors="coerce")
        if metadata[symbol]["instrument_type"] == "CURRENCY":
            # Yahoo FX symbols are already price ratios (for example JPY per USD).
            # Converting the quote currency again would flatten USDJPY=X near 1.
            converted[symbol] = local_price
            continue
        local_price = local_price * unit_factor
        if major == BASE_CURRENCY:
            converted[symbol] = local_price
            continue
        fx_rate = fx_rates[major].reindex(close.index).ffill(limit=5)
        converted[symbol] = local_price * fx_rate

    converted = converted.replace([np.inf, -np.inf], np.nan)
    unusable = [
        symbol for symbol in converted if converted[symbol].dropna().empty
    ]
    if unusable:
        raise PortfolioError(
            "USD conversion produced no usable observations for: "
            + ", ".join(unusable)
        )
    return converted


def download_market_data(
    tickers: tuple[str, ...], benchmark: str, start_date: date
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Download adjusted closes and convert every price series to USD."""
    import yfinance as yf

    symbols = list(dict.fromkeys([*tickers, benchmark.strip().upper()]))
    raw = yf.download(
        symbols,
        start=start_date.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw.empty:
        raise PortfolioError("Market data could not be downloaded. Check the connection and tickers.")
    close = _field_frame(raw, "Close", symbols)
    volume = _field_frame(raw, "Volume", symbols)
    if volume.empty:
        volume = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    missing = [symbol for symbol in symbols if symbol not in close or close[symbol].dropna().empty]
    if missing:
        raise PortfolioError("No usable market data for: " + ", ".join(missing))
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)
    close = close.sort_index().ffill(limit=2)
    metadata = _symbol_metadata(yf, symbols)
    close = _convert_prices_to_usd(close, metadata, yf, start_date)
    # Use one 252-session business-day convention for every supported Yahoo
    # instrument. Weekend moves from continuously traded assets are therefore
    # aggregated into the next business-day return instead of being annualized
    # as extra daily observations in mixed portfolios.
    close = close.loc[close.index.dayofweek < 5].ffill(limit=2)
    volume = volume.reindex(close.index)
    liquidity_proxy_reasons: dict[str, str] = {}
    for symbol in tickers:
        instrument_type = metadata[symbol]["instrument_type"]
        if instrument_type == "FUTURE":
            volume[symbol] = np.nan
            liquidity_proxy_reasons[symbol] = (
                "Conservative proxy — futures contract multiplier unavailable"
            )
        elif instrument_type in {"INDEX", "CURRENCY"}:
            volume[symbol] = np.nan
            liquidity_proxy_reasons[symbol] = (
                f"Conservative proxy — {instrument_type.lower()} volume is not directly tradable ADV"
            )
    assets = close[list(tickers)].dropna(how="any")
    volumes = volume.reindex(index=assets.index, columns=list(tickers)).fillna(0)
    volumes.attrs["liquidity_proxy_reasons"] = liquidity_proxy_reasons
    benchmark_prices = close[benchmark.upper()].dropna()
    if len(assets) < TRADING_DAYS:
        raise PortfolioError(f"Only {len(assets)} common observations are available; at least 252 are required.")
    return assets, benchmark_prices, volumes


def _stable_cholesky(covariance: np.ndarray) -> np.ndarray:
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, 1e-10, None)
    stable = vectors @ np.diag(values) @ vectors.T
    return np.linalg.cholesky((stable + stable.T) / 2 + np.eye(len(stable)) * 1e-12)


def _correlation_covariance(covariance: np.ndarray, correlation_blend: float) -> np.ndarray:
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    volatility = np.sqrt(np.clip(np.diag(covariance), 1e-14, None))
    denominator = np.outer(volatility, volatility)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance, dtype=float),
        where=denominator > 1e-14,
    )
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.clip(correlation, -1.0, 1.0)
    stressed = (1 - correlation_blend) * correlation + correlation_blend * np.ones_like(correlation)
    np.fill_diagonal(stressed, 1.0)
    return np.outer(volatility, volatility) * stressed


def simulate_returns(
    history: pd.DataFrame,
    days: int,
    simulations: int,
    method: str,
    seed: int,
    student_df: int,
    step_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a days × assets × simulations cube and daily regime identifiers."""
    if not np.isfinite(step_scale) or step_scale <= 0:
        raise PortfolioError("Simulation step scale must be positive and finite.")
    rng = np.random.default_rng(seed)
    values = history.to_numpy(dtype=float)
    assets = values.shape[1]
    log_history = np.log1p(np.clip(values, -0.999999, None))
    mean = log_history.mean(axis=0)
    if method == "Historical Bootstrap":
        indices = rng.integers(0, len(values), size=(days, simulations))
        sampled = log_history[indices].transpose(0, 2, 1)
        if not np.isclose(step_scale, 1.0):
            sampled = (
                mean[None, :, None] * step_scale
                + (sampled - mean[None, :, None]) * np.sqrt(step_scale)
            )
        return np.expm1(sampled), np.ones((days, simulations), dtype=np.int8)

    covariance = np.cov(log_history, rowvar=False)
    if method != "Regime Switching":
        chol = _stable_cholesky(covariance)
        normal = rng.standard_normal((days, assets, simulations))
        shocks = (
            np.einsum("ij,djs->dis", chol, normal, optimize=True)
            * np.sqrt(step_scale)
        )
        if method == "Student-t":
            scale = np.sqrt(rng.chisquare(student_df, (days, 1, simulations)) / student_df)
            shocks = shocks / scale * np.sqrt((student_df - 2) / student_df)
        return np.expm1(
            mean[None, :, None] * step_scale + shocks
        ), np.ones((days, simulations), dtype=np.int8)

    transitions = np.array(
        [
            [0.94, 0.04, 0.005, 0.015],
            [0.03, 0.94, 0.015, 0.015],
            [0.00, 0.03, 0.92, 0.05],
            [0.04, 0.06, 0.01, 0.89],
        ]
    )
    mean_shifts = np.array([0.00055, 0.0, -0.0018, 0.0009])
    volatility_multipliers = np.array([0.75, 1.0, 2.5, 1.35])
    correlation_blends = np.array([0.0, 0.0, 0.72, 0.30])
    identity = np.eye(len(transitions))
    if step_scale < 1:
        scaled_transitions = identity + step_scale * (transitions - identity)
    elif step_scale > 1:
        whole_steps = int(np.floor(step_scale))
        fractional_step = step_scale - whole_steps
        scaled_transitions = np.linalg.matrix_power(transitions, whole_steps)
        if fractional_step > 1e-12:
            scaled_transitions = scaled_transitions @ (
                identity + fractional_step * (transitions - identity)
            )
    else:
        scaled_transitions = transitions
    scaled_transitions = np.clip(scaled_transitions, 0.0, None)
    scaled_transitions = scaled_transitions / scaled_transitions.sum(axis=1, keepdims=True)
    cholesky = [
        _stable_cholesky(
            _correlation_covariance(covariance, correlation_blends[state])
            * volatility_multipliers[state] ** 2
            * step_scale
        )
        for state in range(4)
    ]
    regimes = np.empty((days, simulations), dtype=np.int8)
    states = np.ones(simulations, dtype=np.int8)
    output = np.empty((days, assets, simulations), dtype=np.float32)
    for day in range(days):
        if day:
            random_values = rng.random(simulations)
            cumulative = np.cumsum(scaled_transitions[states], axis=1)
            states = (random_values[:, None] > cumulative).sum(axis=1).astype(np.int8)
        regimes[day] = states
        for state in range(4):
            mask = states == state
            count = int(mask.sum())
            if not count:
                continue
            shocks = cholesky[state] @ rng.standard_normal((assets, count))
            state_mean = (mean + mean_shifts[state]) * step_scale
            output[day, :, mask] = np.expm1(state_mean[:, None] + shocks).T
    return output, regimes


def _simulation_step_scale(config: AnalysisConfig) -> float:
    """Return the trading-day equivalent represented by one model step."""
    if config.calendar_days <= 0:
        return 1.0
    intended_trading_days = (
        float(config.calendar_days) * TRADING_DAYS / CALENDAR_DAYS_PER_YEAR
    )
    return max(intended_trading_days / max(config.simulation_days, 1), 1e-8)


def estimate_liquidity(prices: pd.DataFrame, volumes: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate tradability from dollar volume and volatility using transparent proxies."""
    observed_adv = (prices * volumes.reindex(prices.index)).replace(0, np.nan).median()
    observed = observed_adv.notna()
    observed_median = (
        float(observed_adv.dropna().median()) if observed.any() else 10_000_000
    )
    # Instruments without reported volume (often indices, FX, or some funds)
    # receive a conservative, explicitly labelled capacity proxy.
    proxy_adv = max(100_000.0, observed_median * 0.25)
    average_dollar_volume = observed_adv.fillna(proxy_adv).clip(lower=100_000)
    log_adv = np.log10(average_dollar_volume)
    score = 100 / (1 + np.exp(-(log_adv - 7.0)))
    score = score.where(observed, score * 0.65)
    daily_volatility = returns.std().reindex(prices.columns).fillna(0.02)
    proxy_spread_penalty = pd.Series(np.where(observed, 0.0, 15.0), index=prices.columns)
    spread_bps = (
        28 - 0.23 * score + daily_volatility * 120 + proxy_spread_penalty
    ).clip(2, 100)
    participation_capacity = (0.05 * average_dollar_volume).clip(lower=1)
    proxy_reasons = volumes.attrs.get("liquidity_proxy_reasons", {})
    quality = [
        (
            "Observed volume"
            if bool(observed.loc[ticker])
            else proxy_reasons.get(
                ticker, "Conservative proxy — volume unavailable"
            )
        )
        for ticker in prices.columns
    ]
    return pd.DataFrame(
        {
            "Ticker": prices.columns,
            "Liquidity Score": score.to_numpy(),
            "Median Dollar Volume": average_dollar_volume.to_numpy(),
            "Estimated Spread (bps)": spread_bps.to_numpy(),
            "5% ADV Capacity": participation_capacity.to_numpy(),
            "Liquidity Data Quality": quality,
        }
    )


def _maximum_drawdowns(paths: np.ndarray) -> np.ndarray:
    return (paths / np.maximum.accumulate(paths, axis=0) - 1).min(axis=0)


def _time_to_recovery(paths: np.ndarray) -> tuple[np.ndarray, float]:
    """Measure days from each path's deepest trough until its prior peak is regained."""
    simulations = paths.shape[1]
    recovery_days = np.full(simulations, np.nan)
    drawdowns = paths / np.maximum.accumulate(paths, axis=0) - 1
    troughs = np.argmin(drawdowns, axis=0)
    for simulation in range(simulations):
        if float(drawdowns[:, simulation].min()) >= -1e-12:
            recovery_days[simulation] = 0.0
            continue
        trough = int(troughs[simulation])
        prior_peak = float(paths[: trough + 1, simulation].max())
        recovered = np.flatnonzero(paths[trough + 1 :, simulation] >= prior_peak)
        if len(recovered):
            recovery_days[simulation] = recovered[0] + 1
    return recovery_days, float(np.mean(np.isfinite(recovery_days)))


def apply_liquidity_cascade(
    paths: np.ndarray,
    weights: np.ndarray,
    liquidity: pd.DataFrame,
    config: AnalysisConfig,
    terminal_only: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Model forced sales while preserving sale proceeds as investor cash."""
    rng = np.random.default_rng(config.seed + 991)
    adjusted = paths[-1].copy() if terminal_only else paths.copy()
    drawdowns = paths / np.maximum.accumulate(paths, axis=0) - 1
    breached = drawdowns <= config.margin_trigger
    breach_exists = breached.any(axis=0)
    first_breach = np.where(breach_exists, breached.argmax(axis=0), -1)
    redemption = breach_exists & (rng.random(paths.shape[1]) < config.redemption_probability)
    leverage_excess = max(float(config.gross_leverage) - 1.0, 0.0)
    margin_call = (
        breach_exists
        & (config.margin_call_pct > 0)
        & (leverage_excess > 0)
    )
    margin_call_fraction = min(1.0, config.margin_call_pct * leverage_excess)
    required_fraction = (
        redemption * config.redemption_pct
        + margin_call * margin_call_fraction
    )
    sale_fraction = np.clip(
        required_fraction - config.cash_buffer_pct,
        0.0,
        0.99,
    )
    forced_sale = sale_fraction > 0

    spreads = liquidity["Estimated Spread (bps)"].to_numpy() / 10_000
    scores = liquidity["Liquidity Score"].to_numpy() / 100
    average_dollar_volume = liquidity["Median Dollar Volume"].to_numpy()
    capacity_dollars = liquidity["5% ADV Capacity"].to_numpy()
    weighted_score = float(weights @ scores)
    severity = np.where(breach_exists, np.abs(drawdowns[first_breach.clip(min=0), np.arange(paths.shape[1])]), 0)
    liquidity_shortfall = np.zeros(paths.shape[1], dtype=bool)
    sale_cost = np.zeros(paths.shape[1])
    step_scale = _simulation_step_scale(config)
    risk_free_step_growth = (
        1 + max(config.risk_free_rate, -0.99)
    ) ** (step_scale / TRADING_DAYS)

    for simulation in np.flatnonzero(forced_sale):
        day = int(first_breach[simulation])
        current_value = float(paths[day, simulation])
        sale_dollars = current_value * sale_fraction[simulation]
        asset_trade_dollars = sale_dollars * weights
        stress_factor = float(np.clip(1 - 1.6 * severity[simulation], 0.05, 1.0))
        stressed_capacity = capacity_dollars * stress_factor
        liquidity_shortfall[simulation] = bool(
            np.any(asset_trade_dollars > stressed_capacity)
        )
        adv_participation = np.divide(
            asset_trade_dollars,
            average_dollar_volume,
            out=np.zeros_like(asset_trade_dollars),
            where=average_dollar_volume > 0,
        )
        half_spread_rate = float(weights @ spreads / 2)
        slippage_rate = float(0.0015 + 0.0025 * severity[simulation])
        impact_rate = float(
            weights @ (0.012 * np.sqrt(np.clip(adv_participation, 0, 20)))
        )
        cost = min(
            sale_dollars * (half_spread_rate + slippage_rate + impact_rate),
            sale_dollars * 0.50,
        )
        sale_cost[simulation] = cost
        remaining_invested = max(0.0, current_value - sale_dollars)
        cash_proceeds = max(0.0, sale_dollars - cost)
        if terminal_only:
            future_steps = paths.shape[0] - 1 - day
            invested_growth = float(paths[-1, simulation] / current_value)
            cash_growth = risk_free_step_growth**future_steps
            adjusted[simulation] = (
                remaining_invested * invested_growth
                + cash_proceeds * cash_growth
            )
        else:
            relative_path = paths[day:, simulation] / current_value
            cash_growth = risk_free_step_growth ** np.arange(len(relative_path))
            adjusted[day:, simulation] = (
                remaining_invested * relative_path
                + cash_proceeds * cash_growth
            )

    cascade_metrics = {
        "redemption_probability_realized": float(redemption.mean()),
        "margin_call_probability": float(margin_call.mean()),
        "forced_sale_probability": float(forced_sale.mean()),
        "liquidity_shortfall_probability": float(liquidity_shortfall.mean()),
        "expected_forced_sale_cost": float(sale_cost.mean()),
        "conditional_forced_sale_cost": float(sale_cost[forced_sale].mean()) if forced_sale.any() else 0.0,
        "weighted_liquidity_score": weighted_score * 100,
    }
    return adjusted, cascade_metrics


def _asset_betas(returns: pd.DataFrame, benchmark_returns: pd.Series) -> np.ndarray:
    aligned = returns.join(benchmark_returns, how="inner")
    benchmark = aligned.iloc[:, -1]
    benchmark_variance = float(benchmark.var())
    if not np.isfinite(benchmark_variance) or benchmark_variance <= 1e-14:
        raise PortfolioError("Benchmark variance is too small to calculate beta reliably.")
    betas = np.array(
        [aligned[column].cov(benchmark) / benchmark_variance for column in returns.columns],
        dtype=float,
    )
    return np.nan_to_num(betas, nan=0.0, posinf=0.0, neginf=0.0)


def _wealth_with_baseline(returns: pd.Series) -> pd.Series:
    """Compound returns after an explicit 1.0 starting wealth observation."""
    if returns.empty:
        return pd.Series(dtype=float)
    baseline_index = pd.Timestamp(returns.index[0]) - pd.Timedelta(nanoseconds=1)
    baseline = pd.Series([1.0], index=[baseline_index])
    return pd.concat([baseline, (1 + returns).cumprod()])


def stress_test_portfolio(
    tickers: tuple[str, ...],
    weights: np.ndarray,
    betas: np.ndarray,
    liquidity: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Apply transparent stylized shocks inspired by 2008, March 2020, and 2022."""
    scenarios = {
        "2008 Credit Crisis": {"market": -0.37, "liquidity": 0.35, "correlation": 0.85},
        "March 2020 Liquidity Shock": {"market": -0.34, "liquidity": 0.25, "correlation": 0.90},
        "2022 Inflation & Rate Shock": {"market": -0.20, "liquidity": 0.55, "correlation": 0.65},
    }
    bonds = {
        "TLT", "IEF", "SHY", "AGG", "BND", "HYG", "LQD", "TIP",
        "ZB=F", "ZN=F", "ZF=F", "ZT=F",
    }
    gold = {"GLD", "IAU", "SGOL", "GC=F", "MGC=F"}
    energy = {
        "XLE", "XOM", "CVX", "USO", "BNO", "CL=F", "BZ=F", "NG=F",
    }
    scores = liquidity["Liquidity Score"].to_numpy() / 100
    spreads = liquidity["Estimated Spread (bps)"].to_numpy() / 10_000
    rows = []
    for name, scenario in scenarios.items():
        correlation_convergence = scenario["correlation"]
        beta_shocks = betas * scenario["market"]
        shocks = (
            (1 - correlation_convergence) * beta_shocks
            + correlation_convergence * scenario["market"]
        )
        for index, ticker in enumerate(tickers):
            if name.startswith("2008"):
                if ticker in bonds:
                    shocks[index] += 0.08
                if ticker in gold:
                    shocks[index] += 0.05
            elif name.startswith("March 2020"):
                if ticker in bonds:
                    shocks[index] += 0.05
                if ticker in energy:
                    shocks[index] -= 0.22
            else:
                if ticker in bonds:
                    shocks[index] -= 0.14
                if ticker in energy:
                    shocks[index] += 0.22
                if ticker in gold:
                    shocks[index] += 0.01
        shocks = np.clip(shocks, -0.80, 0.45)
        asset_contributions = weights * shocks
        largest_loss_driver = tickers[int(np.argmin(asset_contributions))]
        gross_return = float(weights @ shocks)
        leverage_excess = max(config.gross_leverage - 1.0, 0.0)
        modeled_margin_call = min(1.0, config.margin_call_pct * leverage_excess)
        cash_need = max(
            0.0,
            config.redemption_probability * config.redemption_pct
            + modeled_margin_call
            - config.cash_buffer_pct,
        )
        stressed_pre_sale_value = config.initial_investment * max(
            0.0, 1 + gross_return
        )
        sale_dollars = stressed_pre_sale_value * min(cash_need, 0.99)
        asset_trade_dollars = sale_dollars * weights
        average_dollar_volume = (
            liquidity["Median Dollar Volume"].to_numpy()
            * scenario["liquidity"]
        )
        adv_participation = np.divide(
            asset_trade_dollars,
            average_dollar_volume,
            out=np.zeros_like(asset_trade_dollars),
            where=average_dollar_volume > 0,
        )
        execution_rates = (
            spreads / 2
            + 0.003
            + 0.018 * np.sqrt(np.clip(adv_participation, 0, 20))
        )
        liquidation_cost_dollars = float(asset_trade_dollars @ execution_rates)
        liquidation_cost = liquidation_cost_dollars / config.initial_investment
        net_return = gross_return - liquidation_cost
        rows.append(
            {
                "Scenario": name,
                "Gross Portfolio Shock": gross_return,
                "Liquidation Cost": liquidation_cost,
                "Net Portfolio Shock": net_return,
                "Stressed Value": config.initial_investment * (1 + net_return),
                "Correlation Convergence": correlation_convergence,
                "Liquidity Capacity Multiplier": scenario["liquidity"],
                "Largest Loss Driver": largest_loss_driver,
            }
        )
    return pd.DataFrame(rows)


def resilient_allocation(
    returns: pd.DataFrame, betas: np.ndarray, liquidity: pd.DataFrame
) -> np.ndarray:
    """Construct a transparent crisis-resilient allocation from volatility, beta, and liquidity."""
    volatility = returns.std().to_numpy() * np.sqrt(TRADING_DAYS)
    scores = np.clip(liquidity["Liquidity Score"].to_numpy() / 100, 0.05, 1)
    raw = scores / (np.clip(volatility, 0.01, None) * (1 + np.clip(betas, 0, None)) ** 1.5)
    weights = raw / raw.sum()
    cap = max(0.45, 1 / len(weights))
    for _ in range(20):
        excess = np.maximum(weights - cap, 0)
        if excess.sum() < 1e-8:
            break
        weights = np.minimum(weights, cap)
        eligible = weights < cap - 1e-12
        eligible_total = weights[eligible].sum()
        if not eligible.any() or eligible_total <= 0:
            break
        weights[eligible] += excess.sum() * weights[eligible] / eligible_total
    return weights / weights.sum()


def run_analysis(
    config: AnalysisConfig,
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    volumes: pd.DataFrame,
) -> AnalysisResult:
    """Run historical, regime, liquidity-cascade, stress, and allocation analytics."""
    if config.base_currency.strip().upper() != BASE_CURRENCY:
        raise PortfolioError(
            "This release normalizes all market data to USD; base_currency must be USD."
        )
    if config.initial_investment <= 0:
        raise PortfolioError("Initial investment must be greater than zero.")
    if config.target_value <= config.initial_investment:
        raise PortfolioError("Target portfolio value must be greater than the initial investment.")
    weights = np.asarray(config.weights, dtype=float)
    if len(config.tickers) < 2 or len(weights) != len(config.tickers):
        raise PortfolioError("Portfolio tickers and weights must contain the same assets.")
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, atol=5e-4)
    ):
        raise PortfolioError("Portfolio weights must be finite, non-negative, and total 100%.")
    if config.simulation_days < 1 or config.simulations < 1:
        raise PortfolioError("Simulation horizon and path count must be positive.")
    if not 0 < config.confidence < 1:
        raise PortfolioError("Confidence level must be between 0% and 100%.")
    if config.method == "Student-t" and config.student_df <= 2:
        raise PortfolioError("Student-t degrees of freedom must be greater than two.")
    if config.gross_leverage < 1:
        raise PortfolioError("Collateral-assumption gross leverage cannot be below 1.0×.")
    missing_price_columns = [
        ticker for ticker in config.tickers if ticker not in prices.columns
    ]
    if missing_price_columns:
        raise PortfolioError(
            "Price history is missing portfolio assets: "
            + ", ".join(missing_price_columns)
        )
    returns = (
        prices.loc[:, list(config.tickers)]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    portfolio_returns = returns.dot(weights).rename("Portfolio")
    benchmark_returns = (
        benchmark_prices.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .rename(config.benchmark)
    )
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < TRADING_DAYS:
        raise PortfolioError(
            f"Only {len(aligned)} aligned portfolio/benchmark returns are available; "
            "at least 252 are required."
        )
    portfolio_aligned = aligned.iloc[:, 0]
    benchmark_aligned = aligned.iloc[:, 1]
    cube, regimes = simulate_returns(
        returns,
        config.simulation_days,
        config.simulations,
        config.method,
        config.seed,
        config.student_df,
        _simulation_step_scale(config),
    )
    simulated_portfolio_returns = np.einsum("das,a->ds", cube, weights)
    base_paths = np.vstack(
        [
            np.full(config.simulations, config.initial_investment),
            config.initial_investment * np.cumprod(1 + simulated_portfolio_returns, axis=0),
        ]
    )
    liquidity = estimate_liquidity(prices, volumes, returns)
    paths, cascade_metrics = apply_liquidity_cascade(base_paths, weights, liquidity, config)
    if not np.all(np.isfinite(paths)):
        raise PortfolioError(
            "The selected horizon and assumptions produced non-finite simulated values. "
            "Reduce the horizon or use a less extreme model configuration."
        )
    terminal = paths[-1]
    terminal_returns = terminal / config.initial_investment - 1
    alpha = 1 - config.confidence
    threshold = np.quantile(terminal_returns, alpha)
    tail = terminal_returns[terminal_returns <= threshold]
    maximum_drawdowns = _maximum_drawdowns(paths)
    recovery_steps, recovery_probability = _time_to_recovery(paths)
    calendar_days_per_step = (
        config.calendar_days / config.simulation_days
        if config.calendar_days > 0
        else CALENDAR_DAYS_PER_YEAR / TRADING_DAYS
    )
    recovery_days = recovery_steps * calendar_days_per_step
    annual_return = calendar_time_cagr(portfolio_aligned)
    arithmetic_annual_return = portfolio_aligned.mean() * TRADING_DAYS
    annual_volatility = portfolio_aligned.std() * np.sqrt(TRADING_DAYS)
    if not np.isfinite(annual_volatility) or annual_volatility <= 1e-12:
        raise PortfolioError(
            "Portfolio volatility is too small for risk-adjusted analytics."
        )
    growth = _wealth_with_baseline(portfolio_aligned)
    historical_drawdown = (growth / growth.cummax() - 1).min()
    benchmark_variance = float(benchmark_aligned.var())
    if not np.isfinite(benchmark_variance) or benchmark_variance <= 1e-14:
        raise PortfolioError("Benchmark variance is too small for relative-risk analytics.")
    beta = portfolio_aligned.cov(benchmark_aligned) / benchmark_variance
    benchmark_annual_return = calendar_time_cagr(benchmark_aligned)
    benchmark_arithmetic_return = benchmark_aligned.mean() * TRADING_DAYS
    benchmark_volatility = benchmark_aligned.std() * np.sqrt(TRADING_DAYS)
    if not np.isfinite(benchmark_volatility) or benchmark_volatility <= 1e-12:
        raise PortfolioError(
            "Benchmark volatility is too small for risk-adjusted analytics."
        )
    benchmark_growth = _wealth_with_baseline(benchmark_aligned)
    benchmark_max_drawdown = (benchmark_growth / benchmark_growth.cummax() - 1).min()
    active_returns = portfolio_aligned - benchmark_aligned
    tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS)
    betas = _asset_betas(returns, benchmark_returns)
    covariance = returns.cov().to_numpy() * TRADING_DAYS
    portfolio_volatility = np.sqrt(weights @ covariance @ weights)
    if not np.isfinite(portfolio_volatility) or portfolio_volatility <= 1e-12:
        raise PortfolioError(
            "Portfolio volatility is too small to calculate risk contributions reliably."
        )
    marginal = covariance @ weights / portfolio_volatility
    component = weights * marginal
    contributions = pd.DataFrame(
        {"Ticker": returns.columns, "Weight": weights, "Risk Contribution": component / portfolio_volatility}
    )

    stress_tests = stress_test_portfolio(config.tickers, weights, betas, liquidity, config)
    resilient_weights = resilient_allocation(returns, betas, liquidity)
    resilient_stress = stress_test_portfolio(config.tickers, resilient_weights, betas, liquidity, config)
    comparison_asset_returns = returns.reindex(portfolio_aligned.index).dropna()
    comparison_covariance = (
        comparison_asset_returns.cov().to_numpy() * TRADING_DAYS
    )
    resilient_returns = comparison_asset_returns.dot(resilient_weights)
    resilient_annual_return = calendar_time_cagr(resilient_returns)
    resilient_arithmetic_return = resilient_returns.mean() * TRADING_DAYS
    resilient_volatility = np.sqrt(
        resilient_weights @ comparison_covariance @ resilient_weights
    )
    resilient_sharpe = (
        (resilient_arithmetic_return - config.risk_free_rate) / resilient_volatility
        if resilient_volatility > 0 else np.nan
    )
    resilient_simulated_returns = np.einsum("das,a->ds", cube, resilient_weights)
    resilient_base_paths = np.vstack(
        [
            np.full(config.simulations, config.initial_investment),
            config.initial_investment * np.cumprod(1 + resilient_simulated_returns, axis=0),
        ]
    )
    resilient_terminal, _ = apply_liquidity_cascade(
        resilient_base_paths, resilient_weights, liquidity, config, terminal_only=True
    )
    resilient_target_probability = float(np.mean(resilient_terminal >= config.target_value))
    comparison = pd.DataFrame(
        {
            "Portfolio": ["Current", "Crisis-Resilient"],
            "Historical Return": [
                annual_return,
                resilient_annual_return,
            ],
            "Historical Volatility": [
                annual_volatility,
                resilient_volatility,
            ],
            "Average Stress Loss": [stress_tests["Net Portfolio Shock"].mean(), resilient_stress["Net Portfolio Shock"].mean()],
            "Worst Stress Loss": [stress_tests["Net Portfolio Shock"].min(), resilient_stress["Net Portfolio Shock"].min()],
            "Weighted Liquidity Score": [
                weights @ liquidity["Liquidity Score"].to_numpy(),
                resilient_weights @ liquidity["Liquidity Score"].to_numpy(),
            ],
            "Sharpe Ratio": [
                (arithmetic_annual_return - config.risk_free_rate)
                / annual_volatility,
                resilient_sharpe,
            ],
            "Target Probability": [
                float(np.mean(terminal >= config.target_value)),
                resilient_target_probability,
            ],
        }
    )

    base_correlation = returns.corr().to_numpy()
    off_diagonal = ~np.eye(len(weights), dtype=bool)
    off_diagonal_values = base_correlation[off_diagonal]
    finite_correlations = off_diagonal_values[np.isfinite(off_diagonal_values)]
    normal_corr = (
        float(np.clip(finite_correlations.mean(), -1.0, 1.0))
        if finite_correlations.size
        else 0.0
    )
    crisis_correlation = normal_corr + 0.72 * (1 - normal_corr)
    concentration = (np.sum(weights**2) - 1 / len(weights)) / (1 - 1 / len(weights))
    components = {
        "Drawdown Severity": np.clip(abs(np.median(maximum_drawdowns)) / 0.40 * 100, 0, 100),
        "Liquidity Shortfall": cascade_metrics["liquidity_shortfall_probability"] * 100,
        "Forced-Sale Exposure": cascade_metrics["forced_sale_probability"] * 100,
        "Correlation Instability": np.clip((crisis_correlation - normal_corr) / 0.72 * 100, 0, 100),
        "Concentration": np.clip(concentration * 100, 0, 100),
    }
    component_weights = {
        "Drawdown Severity": 0.30,
        "Liquidity Shortfall": 0.25,
        "Forced-Sale Exposure": 0.20,
        "Correlation Instability": 0.15,
        "Concentration": 0.10,
    }
    fragility_score = sum(components[name] * component_weights[name] for name in components)
    fragility_components = pd.DataFrame(
        {
            "Component": list(components),
            "Risk Score": list(components.values()),
            "Weight": [component_weights[name] for name in components],
            "Weighted Contribution": [components[name] * component_weights[name] for name in components],
        }
    )

    levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    percentiles = pd.DataFrame(
        np.quantile(paths, levels, axis=1).T, columns=["P5", "P25", "Median", "P75", "P95"]
    )
    regime_counts = np.bincount(regimes.ravel(), minlength=4) / regimes.size
    regime_summary = pd.DataFrame(
        {
            "Regime": REGIME_NAMES,
            "Share of Simulated Days": regime_counts,
            "Volatility Multiplier": [0.75, 1.0, 2.5, 1.35],
            "Correlation Blend": [0.0, 0.0, 0.72, 0.30],
        }
    )
    loss_mask = terminal < config.initial_investment
    target_mask = terminal >= config.target_value
    below_target_without_loss_mask = ~(loss_mask | target_mask)
    metrics = {
        "expected_terminal": float(terminal.mean()),
        "median_terminal": float(np.median(terminal)),
        "probability_loss": float(np.mean(loss_mask)),
        "probability_below_target_without_loss": float(np.mean(below_target_without_loss_mask)),
        "probability_target": float(np.mean(target_mask)),
        "var_currency": float(max(0, -threshold * config.initial_investment)),
        "es_currency": float(max(0, -tail.mean() * config.initial_investment)),
        "historical_return": float(annual_return),
        "historical_volatility": float(annual_volatility),
        "sharpe": float(
            (arithmetic_annual_return - config.risk_free_rate) / annual_volatility
        ),
        "historical_max_drawdown": float(historical_drawdown),
        "median_simulated_drawdown": float(np.median(maximum_drawdowns)),
        "beta": float(beta),
        "benchmark_return": float(benchmark_annual_return),
        "benchmark_volatility": float(benchmark_volatility),
        "benchmark_sharpe": float(
            (benchmark_arithmetic_return - config.risk_free_rate)
            / benchmark_volatility
        ),
        "benchmark_max_drawdown": float(benchmark_max_drawdown),
        "benchmark_correlation": float(
            np.nan_to_num(
                portfolio_aligned.corr(benchmark_aligned),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        ),
        "tracking_error": float(tracking_error),
        "information_ratio": float(
            active_returns.mean() * TRADING_DAYS / tracking_error
        ) if tracking_error > 0 else np.nan,
        "p5_terminal": float(np.quantile(terminal, 0.05)),
        "p95_terminal": float(np.quantile(terminal, 0.95)),
        "median_recovery_days": float(np.nanmedian(recovery_days)) if np.isfinite(recovery_days).any() else np.nan,
        "recovery_probability": recovery_probability,
        "normal_correlation": normal_corr,
        "crisis_correlation": crisis_correlation,
        "fragility_score": float(fragility_score),
        **cascade_metrics,
    }
    return AnalysisResult(
        prices,
        returns,
        portfolio_returns,
        benchmark_returns,
        paths,
        terminal,
        percentiles,
        metrics,
        contributions,
        liquidity,
        stress_tests,
        regime_summary,
        fragility_components,
        comparison,
        resilient_weights,
    )
