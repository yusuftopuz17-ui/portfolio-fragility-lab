"""Focused regression tests for drawdown and frontier mathematics."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.analytics import calendar_time_cagr, efficient_frontier, historical_drawdowns


class AnnualizationTests(unittest.TestCase):
    def test_calendar_time_cagr_uses_elapsed_time(self) -> None:
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        total_growth = 1.10
        daily_return = total_growth ** (1 / len(dates)) - 1
        returns = pd.Series(daily_return, index=dates)
        elapsed_years = (
            (dates[-1] - dates[0]).days + 365.2425 / 252
        ) / 365.2425
        expected = total_growth ** (1 / elapsed_years) - 1
        self.assertAlmostEqual(calendar_time_cagr(returns), expected, places=12)


class HistoricalDrawdownTests(unittest.TestCase):
    def test_first_negative_return_is_measured_from_baseline(self) -> None:
        dates = pd.date_range("2024-01-02", periods=3, freq="D")
        result = SimpleNamespace(
            portfolio_returns=pd.Series([-0.10, 0.05, 0.06], index=dates),
            benchmark_returns=pd.Series([0.00, 0.00, 0.00], index=dates),
        )

        drawdowns, episode = historical_drawdowns(result)

        self.assertLess(drawdowns.index[0], dates[0])
        self.assertAlmostEqual(drawdowns.iloc[0]["Portfolio"], 0.0)
        self.assertAlmostEqual(drawdowns.loc[dates[0], "Portfolio"], -0.10)
        self.assertAlmostEqual(episode["maximum_drawdown"], -0.10)
        self.assertEqual(episode["trough"], dates[0])
        self.assertLess(episode["start"], dates[0])
        self.assertEqual(episode["recovery"], dates[-1])


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is required for SLSQP tests")
class EfficientFrontierTests(unittest.TestCase):
    @staticmethod
    def _inputs(asset_count: int):
        rng = np.random.default_rng(7300 + asset_count)
        tickers = tuple(f"A{index}" for index in range(asset_count))
        common = rng.normal(0.00025, 0.007, size=(800, 1))
        idiosyncratic = rng.normal(
            np.linspace(0.00005, 0.00045, asset_count),
            np.linspace(0.004, 0.012, asset_count),
            size=(800, asset_count),
        )
        returns = pd.DataFrame(0.35 * common + idiosyncratic, columns=tickers)
        equal_weights = np.ones(asset_count) / asset_count
        result = SimpleNamespace(
            asset_returns=returns,
            resilient_weights=equal_weights,
        )
        config = SimpleNamespace(
            tickers=tickers,
            weights=equal_weights,
            risk_free_rate=0.02,
            seed=17,
        )
        return result, config

    def test_optimized_weights_respect_constraints_for_two_to_four_assets(self) -> None:
        for asset_count in (2, 3, 4):
            with self.subTest(asset_count=asset_count):
                result, config = self._inputs(asset_count)
                cloud, markers, weights = efficient_frontier(result, config, samples=300)
                cap = max(0.45, 1 / asset_count)

                self.assertIn("Minimum Variance", set(markers["Portfolio"]))
                self.assertIn("Maximum Sharpe", set(markers["Portfolio"]))
                for portfolio in ("Minimum Variance", "Maximum Sharpe"):
                    optimized = weights.loc[weights["Portfolio"] == portfolio, "Weight"]
                    self.assertEqual(len(optimized), asset_count)
                    self.assertAlmostEqual(float(optimized.sum()), 1.0, places=6)
                    self.assertGreaterEqual(float(optimized.min()), -1e-8)
                    self.assertLessEqual(float(optimized.max()), cap + 1e-8)

                marker_index = markers.set_index("Portfolio")
                self.assertLessEqual(
                    marker_index.loc["Minimum Variance", "Annual Volatility"],
                    cloud["Annual Volatility"].min() + 1e-8,
                )
                self.assertGreaterEqual(
                    marker_index.loc["Maximum Sharpe", "Sharpe Ratio"],
                    cloud["Sharpe Ratio"].max() - 1e-8,
                )


if __name__ == "__main__":
    unittest.main()
