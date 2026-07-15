"""Derived portfolio analytics used by the institutional dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252
REGIME_TRANSITION_MATRIX = pd.DataFrame(
    [
        [0.94, 0.04, 0.005, 0.015],
        [0.03, 0.94, 0.015, 0.015],
        [0.00, 0.03, 0.92, 0.05],
        [0.04, 0.06, 0.01, 0.89],
    ],
    index=["Bull", "Normal", "Crisis", "Recovery"],
    columns=["Bull", "Normal", "Crisis", "Recovery"],
)


def correlation_matrices(result) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return historical and crisis-blended asset correlation matrices."""
    normal = result.asset_returns.corr().reindex(index=result.prices.columns, columns=result.prices.columns)
    stressed = 0.28 * normal + 0.72
    np.fill_diagonal(stressed.values, 1.0)
    return normal, stressed


def correlation_interpretation(correlation: pd.DataFrame) -> dict[str, object]:
    """Summarize the strongest and weakest unique correlation pairs."""
    mask = np.triu(np.ones(correlation.shape, dtype=bool), k=1)
    pairs = correlation.where(mask).stack()
    if pairs.empty:
        return {"highest_pair": "N/A", "highest": np.nan, "lowest_pair": "N/A", "lowest": np.nan}
    high_pair = pairs.idxmax()
    low_pair = pairs.idxmin()
    return {
        "highest_pair": f"{high_pair[0]} / {high_pair[1]}",
        "highest": float(pairs.loc[high_pair]),
        "lowest_pair": f"{low_pair[0]} / {low_pair[1]}",
        "lowest": float(pairs.loc[low_pair]),
    }


def rolling_analytics(result, risk_free_rate: float) -> pd.DataFrame:
    """Calculate rolling volatility, Sharpe, beta, and benchmark correlation."""
    aligned = pd.concat(
        [result.portfolio_returns.rename("Portfolio"), result.benchmark_returns.rename("Benchmark")],
        axis=1,
    ).dropna()
    rolling_volatility = aligned["Portfolio"].rolling(63).std() * np.sqrt(TRADING_DAYS)
    rolling_return = aligned["Portfolio"].rolling(126).mean() * TRADING_DAYS
    rolling_std = aligned["Portfolio"].rolling(126).std() * np.sqrt(TRADING_DAYS)
    rolling_sharpe = (rolling_return - risk_free_rate) / rolling_std.replace(0, np.nan)
    benchmark_variance = aligned["Benchmark"].rolling(126).var()
    rolling_beta = (
        aligned["Portfolio"].rolling(126).cov(aligned["Benchmark"])
        / benchmark_variance.replace(0, np.nan)
    )
    rolling_correlation = aligned["Portfolio"].rolling(126).corr(aligned["Benchmark"])
    return pd.DataFrame(
        {
            "Rolling 63D Volatility": rolling_volatility,
            "Rolling 126D Sharpe": rolling_sharpe,
            "Rolling 126D Beta": rolling_beta,
            "Rolling 126D Correlation": rolling_correlation,
        }
    ).dropna(how="all")


def historical_drawdowns(result) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return portfolio/benchmark drawdown paths and the portfolio drawdown episode."""
    aligned = pd.concat(
        [result.portfolio_returns.rename("Portfolio"), result.benchmark_returns.rename("Benchmark")],
        axis=1,
    ).dropna()
    growth = (1 + aligned).cumprod()
    drawdowns = growth.div(growth.cummax()).sub(1)
    trough = drawdowns["Portfolio"].idxmin()
    start = growth.loc[:trough, "Portfolio"].idxmax()
    prior_peak = growth.loc[start, "Portfolio"]
    recovered = growth.loc[trough:, "Portfolio"] >= prior_peak
    recovery = recovered[recovered].index[0] if recovered.any() else None
    end = recovery if recovery is not None else growth.index[-1]
    return drawdowns, {
        "maximum_drawdown": float(drawdowns.loc[trough, "Portfolio"]),
        "start": start,
        "trough": trough,
        "recovery": recovery,
        "duration_days": int((end - start).days),
    }


def risk_contribution_table(result) -> pd.DataFrame:
    """Enrich risk contributions with concentration gaps."""
    table = result.risk_contributions.copy()
    total = table["Risk Contribution"].sum()
    if np.isfinite(total) and abs(total) > 1e-12:
        table["Risk Contribution"] = table["Risk Contribution"] / total
    table["Risk Concentration Gap"] = table["Risk Contribution"] - table["Weight"]
    table["Flag"] = np.where(table["Risk Concentration Gap"] > 0.05, "Above weight", "Balanced")
    return table


def _portfolio_point(weights: np.ndarray, means: np.ndarray, covariance: np.ndarray, rf: float) -> tuple[float, float, float]:
    annual_return = float(weights @ means)
    volatility = float(np.sqrt(max(weights @ covariance @ weights, 0)))
    sharpe = (annual_return - rf) / volatility if volatility > 0 else np.nan
    return annual_return, volatility, sharpe


def efficient_frontier(result, config, samples: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a deterministic long-only frontier cloud with a 45% position cap."""
    returns = result.asset_returns.reindex(columns=config.tickers)
    means = returns.mean().to_numpy() * TRADING_DAYS
    covariance = returns.cov().to_numpy() * TRADING_DAYS
    assets = len(config.tickers)
    rng = np.random.default_rng(config.seed + 404)
    accepted: list[np.ndarray] = []
    needed = samples
    for _ in range(12):
        candidates = rng.dirichlet(np.ones(assets) * 1.3, size=max(needed * 3, 1000))
        valid = candidates[candidates.max(axis=1) <= 0.45 + 1e-12]
        if len(valid):
            accepted.append(valid[:needed])
            needed -= min(len(valid), needed)
        if needed <= 0:
            break
    weights_cloud = np.vstack(accepted) if accepted else np.tile(np.ones(assets) / assets, (samples, 1))
    weights_cloud = weights_cloud[:samples]
    frontier_return = weights_cloud @ means
    frontier_variance = np.einsum("ij,jk,ik->i", weights_cloud, covariance, weights_cloud)
    frontier_volatility = np.sqrt(np.clip(frontier_variance, 0, None))
    frontier_sharpe = np.divide(
        frontier_return - config.risk_free_rate,
        frontier_volatility,
        out=np.full_like(frontier_return, np.nan),
        where=frontier_volatility > 0,
    )
    cloud = pd.DataFrame(
        {"Annual Return": frontier_return, "Annual Volatility": frontier_volatility, "Sharpe Ratio": frontier_sharpe}
    )
    min_index = int(np.nanargmin(frontier_volatility))
    max_index = int(np.nanargmax(frontier_sharpe))
    named_weights = {
        "Current": np.asarray(config.weights, dtype=float),
        "Minimum Variance": weights_cloud[min_index],
        "Maximum Sharpe": weights_cloud[max_index],
        "Equal Weight": np.ones(assets) / assets,
        "Crisis-Resilient": np.asarray(result.resilient_weights, dtype=float),
    }
    marker_rows = []
    weight_rows = []
    for name, weights in named_weights.items():
        annual_return, volatility, sharpe = _portfolio_point(weights, means, covariance, config.risk_free_rate)
        marker_rows.append(
            {"Portfolio": name, "Annual Return": annual_return, "Annual Volatility": volatility, "Sharpe Ratio": sharpe}
        )
        for ticker, weight in zip(config.tickers, weights):
            weight_rows.append({"Portfolio": name, "Ticker": ticker, "Weight": float(weight)})
    return cloud, pd.DataFrame(marker_rows), pd.DataFrame(weight_rows)


def simulation_statistics(result, config) -> pd.DataFrame:
    """Return a compact, consistently formatted terminal distribution dataset."""
    values = pd.Series(result.terminal_values, dtype=float)
    returns = values / config.initial_investment - 1
    metrics = result.metrics
    return pd.DataFrame(
        {
            "Statistic": [
                "Mean", "Median", "Standard deviation", "Skewness", "Excess kurtosis",
                "5th percentile", "95th percentile", "Value at Risk", "Expected Shortfall",
                "Probability of loss", "Probability of reaching target",
            ],
            "Value": [
                values.mean(), values.median(), values.std(), returns.skew(), returns.kurt(),
                values.quantile(0.05), values.quantile(0.95), metrics["var_currency"], metrics["es_currency"],
                metrics["probability_loss"], metrics["probability_target"],
            ],
            "Format": ["currency", "currency", "currency", "ratio", "ratio", "currency", "currency", "currency", "currency", "probability", "probability"],
        }
    )


def regime_assumptions(result) -> pd.DataFrame:
    """Augment regime outputs with transparent model assumptions."""
    table = result.regime_summary.copy()
    table["Return Adjustment"] = [0.00055, 0.0, -0.0018, 0.0009]
    table["Average Duration (days)"] = [
        1 / (1 - REGIME_TRANSITION_MATRIX.loc[name, name]) for name in table["Regime"]
    ]
    return table
