"""Reusable portfolio analytics and Monte Carlo engine for the Streamlit app."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


TRADING_DAYS = 252


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


def validate_portfolio(table: pd.DataFrame) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Validate an editable ticker/weight table and normalize percentage weights."""
    data = table.copy().dropna(how="all")
    data["Ticker"] = data["Ticker"].astype(str).str.strip().str.upper()
    data = data[data["Ticker"].ne("") & data["Ticker"].ne("NAN")]
    if len(data) < 2:
        raise PortfolioError("En az iki farklı varlık eklemelisin.")
    if data["Ticker"].duplicated().any():
        raise PortfolioError("Aynı ticker birden fazla kez kullanılamaz.")
    weights = pd.to_numeric(data["Weight (%)"], errors="coerce")
    if weights.isna().any() or (weights < 0).any() or weights.sum() <= 0:
        raise PortfolioError("Ağırlıklar pozitif sayılar olmalıdır.")
    if not np.isclose(weights.sum(), 100, atol=0.05):
        raise PortfolioError(f"Ağırlıkların toplamı %100 olmalı. Şu an: %{weights.sum():.2f}")
    return tuple(data["Ticker"]), tuple((weights / 100).to_numpy())


def download_prices(tickers: tuple[str, ...], benchmark: str, start_date: date) -> tuple[pd.DataFrame, pd.Series]:
    """Download and conservatively align auto-adjusted Yahoo Finance closes."""
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
        raise PortfolioError("Piyasa verisi indirilemedi. İnternet bağlantısını ve ticker'ları kontrol et.")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
    close.columns = [str(column).upper() for column in close.columns]
    missing = [symbol for symbol in symbols if symbol not in close or close[symbol].dropna().empty]
    if missing:
        raise PortfolioError("Veri bulunamayan ticker: " + ", ".join(missing))
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index().ffill(limit=2)
    assets = close[list(tickers)].dropna(how="any")
    benchmark_prices = close[benchmark.upper()].dropna()
    if len(assets) < TRADING_DAYS:
        raise PortfolioError(f"Ortak tarih aralığında yalnızca {len(assets)} gözlem var; en az 252 gerekli.")
    return assets, benchmark_prices


def _stable_cholesky(covariance: np.ndarray) -> np.ndarray:
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, 1e-10, None)
    stable = vectors @ np.diag(values) @ vectors.T
    return np.linalg.cholesky((stable + stable.T) / 2 + np.eye(len(stable)) * 1e-12)


def simulate_returns(
    history: pd.DataFrame,
    days: int,
    simulations: int,
    method: str,
    seed: int,
    student_df: int,
) -> np.ndarray:
    """Return a days × assets × simulations vectorized return cube."""
    rng = np.random.default_rng(seed)
    values = history.to_numpy(dtype=float)
    assets = values.shape[1]
    if method == "Historical Bootstrap":
        indices = rng.integers(0, len(values), size=(days, simulations))
        return values[indices].transpose(0, 2, 1)
    log_history = np.log1p(np.clip(values, -0.999999, None))
    mean = log_history.mean(axis=0)
    chol = _stable_cholesky(np.cov(log_history, rowvar=False))
    normal = rng.standard_normal((days, assets, simulations))
    shocks = np.einsum("ij,djs->dis", chol, normal, optimize=True)
    if method == "Student-t":
        scale = np.sqrt(rng.chisquare(student_df, (days, 1, simulations)) / student_df)
        shocks = shocks / scale * np.sqrt((student_df - 2) / student_df)
    return np.expm1(mean[None, :, None] + shocks)


def _maximum_drawdowns(paths: np.ndarray) -> np.ndarray:
    return (paths / np.maximum.accumulate(paths, axis=0) - 1).min(axis=0)


def run_analysis(config: AnalysisConfig, prices: pd.DataFrame, benchmark_prices: pd.Series) -> AnalysisResult:
    """Run historical analytics, simulation, downside risk, and risk decomposition."""
    returns = prices.pct_change(fill_method=None).dropna()
    weights = np.asarray(config.weights)
    portfolio_returns = returns.dot(weights).rename("Portfolio")
    benchmark_returns = benchmark_prices.pct_change(fill_method=None).dropna().rename(config.benchmark)
    cube = simulate_returns(
        returns,
        config.simulation_days,
        config.simulations,
        config.method,
        config.seed,
        config.student_df,
    )
    simulated_portfolio_returns = np.einsum("das,a->ds", cube, weights)
    paths = np.vstack(
        [
            np.full(config.simulations, config.initial_investment),
            config.initial_investment * np.cumprod(1 + simulated_portfolio_returns, axis=0),
        ]
    )
    terminal = paths[-1]
    terminal_returns = terminal / config.initial_investment - 1
    alpha = 1 - config.confidence
    threshold = np.quantile(terminal_returns, alpha)
    tail = terminal_returns[terminal_returns <= threshold]
    maximum_drawdowns = _maximum_drawdowns(paths)
    annual_return = (1 + portfolio_returns).prod() ** (TRADING_DAYS / len(portfolio_returns)) - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
    growth = (1 + portfolio_returns).cumprod()
    historical_drawdown = (growth / growth.cummax() - 1).min()
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    beta = aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var()
    covariance = returns.cov().to_numpy() * TRADING_DAYS
    portfolio_volatility = np.sqrt(weights @ covariance @ weights)
    marginal = covariance @ weights / portfolio_volatility
    component = weights * marginal
    contributions = pd.DataFrame(
        {
            "Ticker": returns.columns,
            "Weight": weights,
            "Risk Contribution": component / portfolio_volatility,
        }
    )
    levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    percentiles = pd.DataFrame(
        np.quantile(paths, levels, axis=1).T,
        columns=["P5", "P25", "Median", "P75", "P95"],
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
        "p5_terminal": float(np.quantile(terminal, 0.05)),
        "p95_terminal": float(np.quantile(terminal, 0.95)),
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
    )
