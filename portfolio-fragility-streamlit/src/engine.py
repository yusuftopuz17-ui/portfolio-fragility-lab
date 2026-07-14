"""Portfolio simulation, regime risk, liquidity cascades, and stress testing."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


TRADING_DAYS = 252
REGIME_NAMES = ("Bull", "Normal", "Crisis", "Recovery")


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
    weights = pd.to_numeric(data["Weight (%)"], errors="coerce")
    if weights.isna().any() or (weights < 0).any() or weights.sum() <= 0:
        raise PortfolioError("Weights must be non-negative numbers.")
    if not np.isclose(weights.sum(), 100, atol=0.05):
        raise PortfolioError(f"Weights must total 100%. Current total: {weights.sum():.2f}%")
    return tuple(data["Ticker"]), tuple((weights / 100).to_numpy())


def download_market_data(
    tickers: tuple[str, ...], benchmark: str, start_date: date
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Download adjusted closes and volumes, then align assets conservatively."""
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
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy() if "Volume" in raw.columns.get_level_values(0) else close * np.nan
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": symbols[0]})
    close.columns = [str(column).upper() for column in close.columns]
    volume.columns = [str(column).upper() for column in volume.columns]
    missing = [symbol for symbol in symbols if symbol not in close or close[symbol].dropna().empty]
    if missing:
        raise PortfolioError("No usable market data for: " + ", ".join(missing))
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)
    close = close.sort_index().ffill(limit=2)
    assets = close[list(tickers)].dropna(how="any")
    volumes = volume.reindex(assets.index)[list(tickers)].fillna(0)
    benchmark_prices = close[benchmark.upper()].dropna()
    if len(assets) < TRADING_DAYS:
        raise PortfolioError(f"Only {len(assets)} common observations are available; at least 252 are required.")
    return assets, benchmark_prices, volumes


def _stable_cholesky(covariance: np.ndarray) -> np.ndarray:
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, 1e-10, None)
    stable = vectors @ np.diag(values) @ vectors.T
    return np.linalg.cholesky((stable + stable.T) / 2 + np.eye(len(stable)) * 1e-12)


def _correlation_covariance(covariance: np.ndarray, correlation_blend: float) -> np.ndarray:
    volatility = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(volatility, volatility)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return a days × assets × simulations cube and daily regime identifiers."""
    rng = np.random.default_rng(seed)
    values = history.to_numpy(dtype=float)
    assets = values.shape[1]
    if method == "Historical Bootstrap":
        indices = rng.integers(0, len(values), size=(days, simulations))
        return values[indices].transpose(0, 2, 1), np.ones((days, simulations), dtype=np.int8)

    log_history = np.log1p(np.clip(values, -0.999999, None))
    mean = log_history.mean(axis=0)
    covariance = np.cov(log_history, rowvar=False)
    if method != "Regime Switching":
        chol = _stable_cholesky(covariance)
        normal = rng.standard_normal((days, assets, simulations))
        shocks = np.einsum("ij,djs->dis", chol, normal, optimize=True)
        if method == "Student-t":
            scale = np.sqrt(rng.chisquare(student_df, (days, 1, simulations)) / student_df)
            shocks = shocks / scale * np.sqrt((student_df - 2) / student_df)
        return np.expm1(mean[None, :, None] + shocks), np.ones((days, simulations), dtype=np.int8)

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
    cholesky = [
        _stable_cholesky(
            _correlation_covariance(covariance, correlation_blends[state])
            * volatility_multipliers[state] ** 2
        )
        for state in range(4)
    ]
    regimes = np.empty((days, simulations), dtype=np.int8)
    states = np.ones(simulations, dtype=np.int8)
    output = np.empty((days, assets, simulations), dtype=np.float32)
    for day in range(days):
        if day:
            random_values = rng.random(simulations)
            cumulative = np.cumsum(transitions[states], axis=1)
            states = (random_values[:, None] > cumulative).sum(axis=1).astype(np.int8)
        regimes[day] = states
        for state in range(4):
            mask = states == state
            count = int(mask.sum())
            if not count:
                continue
            shocks = cholesky[state] @ rng.standard_normal((assets, count))
            state_mean = mean + mean_shifts[state]
            output[day, :, mask] = np.expm1(state_mean[:, None] + shocks).T
    return output, regimes


def estimate_liquidity(prices: pd.DataFrame, volumes: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate tradability from dollar volume and volatility using transparent proxies."""
    average_dollar_volume = (prices * volumes.reindex(prices.index)).replace(0, np.nan).median()
    fallback = float(average_dollar_volume.dropna().median()) if average_dollar_volume.notna().any() else 10_000_000
    average_dollar_volume = average_dollar_volume.fillna(fallback).clip(lower=100_000)
    log_adv = np.log10(average_dollar_volume)
    score = 100 / (1 + np.exp(-(log_adv - 7.0)))
    daily_volatility = returns.std().reindex(prices.columns).fillna(0.02)
    spread_bps = (28 - 0.23 * score + daily_volatility * 120).clip(2, 80)
    participation_capacity = (0.05 * average_dollar_volume).clip(lower=1)
    return pd.DataFrame(
        {
            "Ticker": prices.columns,
            "Liquidity Score": score.to_numpy(),
            "Median Dollar Volume": average_dollar_volume.to_numpy(),
            "Estimated Spread (bps)": spread_bps.to_numpy(),
            "5% ADV Capacity": participation_capacity.to_numpy(),
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
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply redemption, margin-call, forced-sale, spread, slippage, and market-impact losses."""
    rng = np.random.default_rng(config.seed + 991)
    adjusted = paths.copy()
    drawdowns = paths / np.maximum.accumulate(paths, axis=0) - 1
    breached = drawdowns <= config.margin_trigger
    breach_exists = breached.any(axis=0)
    first_breach = np.where(breach_exists, breached.argmax(axis=0), -1)
    redemption = breach_exists & (rng.random(paths.shape[1]) < config.redemption_probability)
    margin_call = breach_exists & (config.margin_call_pct > 0)
    required_fraction = redemption * config.redemption_pct + margin_call * config.margin_call_pct
    sale_fraction = np.maximum(0.0, required_fraction - config.cash_buffer_pct)
    forced_sale = sale_fraction > 0

    scores = liquidity["Liquidity Score"].to_numpy() / 100
    spreads = liquidity["Estimated Spread (bps)"].to_numpy() / 10_000
    weighted_score = float(weights @ scores)
    weighted_half_spread = float(weights @ spreads / 2)
    severity = np.where(breach_exists, np.abs(drawdowns[first_breach.clip(min=0), np.arange(paths.shape[1])]), 0)
    capacity_fraction = np.clip(0.30 * weighted_score * (1 - 1.6 * severity), 0.01, 0.30)
    liquidity_shortfall = forced_sale & (sale_fraction > capacity_fraction)
    participation = np.divide(sale_fraction, capacity_fraction, out=np.zeros_like(sale_fraction), where=capacity_fraction > 0)
    slippage = 0.0015 + 0.0025 * severity
    market_impact = 0.012 * np.sqrt(np.clip(participation, 0, 10))
    cost_rate = weighted_half_spread + slippage + market_impact
    sale_cost = np.zeros(paths.shape[1])

    for simulation in np.flatnonzero(forced_sale):
        day = int(first_breach[simulation])
        current_value = adjusted[day, simulation]
        withdrawal = current_value * sale_fraction[simulation]
        cost = withdrawal * cost_rate[simulation]
        sale_cost[simulation] = cost
        multiplier = max(0.01, (current_value - withdrawal - cost) / current_value)
        adjusted[day:, simulation] *= multiplier

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
    return np.array([aligned[column].cov(benchmark) / benchmark.var() for column in returns.columns])


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
    bonds = {"TLT", "IEF", "SHY", "AGG", "BND", "HYG", "LQD", "TIP"}
    gold = {"GLD", "IAU", "SGOL"}
    energy = {"XLE", "XOM", "CVX", "USO", "BNO"}
    scores = liquidity["Liquidity Score"].to_numpy() / 100
    spreads = liquidity["Estimated Spread (bps)"].to_numpy() / 10_000
    rows = []
    for name, scenario in scenarios.items():
        shocks = betas * scenario["market"]
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
        gross_return = float(weights @ shocks)
        stressed_capacity = max(0.01, 0.30 * float(weights @ scores) * scenario["liquidity"])
        cash_need = max(0.0, config.redemption_pct + config.margin_call_pct - config.cash_buffer_pct)
        participation = cash_need / stressed_capacity
        liquidation_cost = cash_need * (
            float(weights @ spreads / 2) + 0.003 + 0.018 * np.sqrt(np.clip(participation, 0, 10))
        )
        net_return = gross_return - liquidation_cost
        rows.append(
            {
                "Scenario": name,
                "Gross Portfolio Shock": gross_return,
                "Liquidation Cost": liquidation_cost,
                "Net Portfolio Shock": net_return,
                "Stressed Value": config.initial_investment * (1 + net_return),
                "Correlation Assumption": scenario["correlation"],
                "Liquidity Capacity Multiplier": scenario["liquidity"],
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
    for _ in range(8):
        excess = np.maximum(weights - 0.45, 0)
        if excess.sum() < 1e-8:
            break
        weights = np.minimum(weights, 0.45)
        eligible = weights < 0.45
        weights[eligible] += excess.sum() * weights[eligible] / weights[eligible].sum()
    return weights / weights.sum()


def run_analysis(
    config: AnalysisConfig,
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    volumes: pd.DataFrame,
) -> AnalysisResult:
    """Run historical, regime, liquidity-cascade, stress, and allocation analytics."""
    returns = prices.pct_change(fill_method=None).dropna()
    weights = np.asarray(config.weights)
    portfolio_returns = returns.dot(weights).rename("Portfolio")
    benchmark_returns = benchmark_prices.pct_change(fill_method=None).dropna().rename(config.benchmark)
    cube, regimes = simulate_returns(
        returns, config.simulation_days, config.simulations, config.method, config.seed, config.student_df
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
    terminal = paths[-1]
    terminal_returns = terminal / config.initial_investment - 1
    alpha = 1 - config.confidence
    threshold = np.quantile(terminal_returns, alpha)
    tail = terminal_returns[terminal_returns <= threshold]
    maximum_drawdowns = _maximum_drawdowns(paths)
    recovery_days, recovery_probability = _time_to_recovery(paths)
    annual_return = (1 + portfolio_returns).prod() ** (TRADING_DAYS / len(portfolio_returns)) - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
    growth = (1 + portfolio_returns).cumprod()
    historical_drawdown = (growth / growth.cummax() - 1).min()
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    beta = aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var()
    benchmark_aligned = aligned.iloc[:, 1]
    benchmark_annual_return = (
        (1 + benchmark_aligned).prod() ** (TRADING_DAYS / len(benchmark_aligned)) - 1
    )
    benchmark_volatility = benchmark_aligned.std() * np.sqrt(TRADING_DAYS)
    benchmark_growth = (1 + benchmark_aligned).cumprod()
    benchmark_max_drawdown = (benchmark_growth / benchmark_growth.cummax() - 1).min()
    active_returns = aligned.iloc[:, 0] - benchmark_aligned
    tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS)
    betas = _asset_betas(returns, benchmark_returns)
    covariance = returns.cov().to_numpy() * TRADING_DAYS
    portfolio_volatility = np.sqrt(weights @ covariance @ weights)
    marginal = covariance @ weights / portfolio_volatility
    component = weights * marginal
    contributions = pd.DataFrame(
        {"Ticker": returns.columns, "Weight": weights, "Risk Contribution": component / portfolio_volatility}
    )

    stress_tests = stress_test_portfolio(config.tickers, weights, betas, liquidity, config)
    resilient_weights = resilient_allocation(returns, betas, liquidity)
    resilient_stress = stress_test_portfolio(config.tickers, resilient_weights, betas, liquidity, config)
    comparison = pd.DataFrame(
        {
            "Portfolio": ["Current", "Crisis-Resilient"],
            "Historical Return": [returns.dot(weights).mean() * TRADING_DAYS, returns.dot(resilient_weights).mean() * TRADING_DAYS],
            "Historical Volatility": [
                np.sqrt(weights @ covariance @ weights),
                np.sqrt(resilient_weights @ covariance @ resilient_weights),
            ],
            "Average Stress Loss": [stress_tests["Net Portfolio Shock"].mean(), resilient_stress["Net Portfolio Shock"].mean()],
            "Worst Stress Loss": [stress_tests["Net Portfolio Shock"].min(), resilient_stress["Net Portfolio Shock"].min()],
            "Weighted Liquidity Score": [
                weights @ liquidity["Liquidity Score"].to_numpy(),
                resilient_weights @ liquidity["Liquidity Score"].to_numpy(),
            ],
        }
    )

    base_correlation = returns.corr().to_numpy()
    off_diagonal = ~np.eye(len(weights), dtype=bool)
    normal_corr = float(base_correlation[off_diagonal].mean()) if len(weights) > 1 else 0
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
    metrics = {
        "expected_terminal": float(terminal.mean()),
        "median_terminal": float(np.median(terminal)),
        "probability_loss": float(np.mean(terminal < config.initial_investment)),
        "probability_target": float(np.mean(terminal >= config.target_value)),
        "var_currency": float(max(0, -threshold * config.initial_investment)),
        "es_currency": float(max(0, -tail.mean() * config.initial_investment)),
        "historical_return": float(annual_return),
        "historical_volatility": float(annual_volatility),
        "sharpe": float((annual_return - config.risk_free_rate) / annual_volatility),
        "historical_max_drawdown": float(historical_drawdown),
        "median_simulated_drawdown": float(np.median(maximum_drawdowns)),
        "beta": float(beta),
        "benchmark_return": float(benchmark_annual_return),
        "benchmark_volatility": float(benchmark_volatility),
        "benchmark_sharpe": float(
            (benchmark_annual_return - config.risk_free_rate) / benchmark_volatility
        ),
        "benchmark_max_drawdown": float(benchmark_max_drawdown),
        "benchmark_correlation": float(aligned.iloc[:, 0].corr(benchmark_aligned)),
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
