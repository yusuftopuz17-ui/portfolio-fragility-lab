"""Regression checks for portfolio scale, liquidity, and risk identities."""

from __future__ import annotations

from datetime import date
import unittest

import numpy as np
import pandas as pd

from src.engine import (
    AnalysisConfig,
    _simulation_step_scale,
    _time_to_recovery,
    apply_liquidity_cascade,
    run_analysis,
    simulate_returns,
)


def _config(initial: float = 500.0, simulations: int = 128) -> AnalysisConfig:
    return AnalysisConfig(
        tickers=("AAA", "BBB"),
        weights=(0.55, 0.45),
        benchmark="BMK",
        start_date=date(2020, 1, 1),
        initial_investment=initial,
        simulation_days=63,
        simulations=simulations,
        confidence=0.95,
        target_value=initial * 1.5,
        risk_free_rate=0.03,
        method="Historical Bootstrap",
        seed=7,
        redemption_probability=0.4,
        redemption_pct=0.15,
        margin_call_pct=0.08,
        margin_trigger=-0.10,
        cash_buffer_pct=0.02,
        gross_leverage=1.0,
    )


def _liquidity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Liquidity Score": [80.0, 60.0],
            "Median Dollar Volume": [1_000_000.0, 500_000.0],
            "Estimated Spread (bps)": [10.0, 20.0],
            "5% ADV Capacity": [50_000.0, 25_000.0],
        }
    )


class LiquidityCascadeTests(unittest.TestCase):
    def test_unlevered_portfolio_has_no_margin_call(self) -> None:
        config = _config()
        paths = np.array(
            [
                [500.0, 500.0],
                [400.0, 400.0],
                [400.0, 400.0],
            ]
        )
        _, metrics = apply_liquidity_cascade(
            paths,
            np.asarray(config.weights),
            _liquidity(),
            config,
        )
        self.assertEqual(metrics["margin_call_probability"], 0.0)

    def test_forced_sale_keeps_proceeds_as_cash(self) -> None:
        config = _config(simulations=2)
        config = AnalysisConfig(
            **{
                **config.__dict__,
                "redemption_probability": 1.0,
                "redemption_pct": 0.50,
                "cash_buffer_pct": 0.0,
                "risk_free_rate": 0.0,
            }
        )
        paths = np.array(
            [
                [500.0, 500.0],
                [400.0, 400.0],
                [400.0, 400.0],
            ]
        )
        adjusted, metrics = apply_liquidity_cascade(
            paths,
            np.asarray(config.weights),
            _liquidity(),
            config,
        )
        self.assertTrue(np.all(adjusted[-1] > 390.0))
        self.assertTrue(np.all(adjusted[-1] < 400.0))
        self.assertGreater(metrics["expected_forced_sale_cost"], 0.0)

    def test_liquidity_cost_scales_with_portfolio_size(self) -> None:
        small_config = _config(initial=500.0, simulations=2)
        large_config = _config(initial=500_000_000.0, simulations=2)
        common = {
            "redemption_probability": 1.0,
            "redemption_pct": 0.50,
            "cash_buffer_pct": 0.0,
            "risk_free_rate": 0.0,
        }
        small_config = AnalysisConfig(**{**small_config.__dict__, **common})
        large_config = AnalysisConfig(**{**large_config.__dict__, **common})
        small_paths = np.array([[500.0] * 2, [400.0] * 2, [400.0] * 2])
        large_paths = small_paths * 1_000_000
        _, small = apply_liquidity_cascade(
            small_paths, np.asarray(small_config.weights), _liquidity(), small_config
        )
        _, large = apply_liquidity_cascade(
            large_paths, np.asarray(large_config.weights), _liquidity(), large_config
        )
        self.assertEqual(small["liquidity_shortfall_probability"], 0.0)
        self.assertEqual(large["liquidity_shortfall_probability"], 1.0)
        self.assertGreater(
            large["expected_forced_sale_cost"] / large_config.initial_investment,
            small["expected_forced_sale_cost"] / small_config.initial_investment,
        )


class HorizonScalingTests(unittest.TestCase):
    def test_hour_horizon_uses_fractional_trading_day(self) -> None:
        config = AnalysisConfig(
            **{
                **_config().__dict__,
                "simulation_days": 1,
                "calendar_days": 1 / 24,
            }
        )
        expected = (1 / 24) * 252 / 365.2425
        self.assertAlmostEqual(_simulation_step_scale(config), expected, places=12)

    def test_fractional_step_scales_diffusion_volatility(self) -> None:
        rng = np.random.default_rng(101)
        history = pd.DataFrame(
            {"AAA": rng.normal(0.0003, 0.012, 2_000)}
        )
        full, _ = simulate_returns(
            history, 1, 20_000, "Geometric Brownian Motion", 22, 5, 1.0
        )
        hourly_scale = (1 / 24) * 252 / 365.2425
        hourly, _ = simulate_returns(
            history,
            1,
            20_000,
            "Geometric Brownian Motion",
            22,
            5,
            hourly_scale,
        )
        ratio = np.log1p(hourly[0, 0]).std() / np.log1p(full[0, 0]).std()
        self.assertAlmostEqual(ratio, np.sqrt(hourly_scale), places=10)

    def test_no_drawdown_path_has_zero_recovery_time(self) -> None:
        paths = np.array([[100.0, 100.0], [101.0, 90.0], [102.0, 100.0]])
        recovery_days, probability = _time_to_recovery(paths)
        self.assertEqual(recovery_days[0], 0.0)
        self.assertEqual(recovery_days[1], 1.0)
        self.assertEqual(probability, 1.0)


class FullAnalysisInvariantTests(unittest.TestCase):
    def test_analysis_respects_initial_scale_and_probability_identities(self) -> None:
        rng = np.random.default_rng(29)
        index = pd.bdate_range("2022-01-03", periods=520)
        asset_returns = rng.multivariate_normal(
            [0.00035, 0.00020],
            [[0.00012, 0.000035], [0.000035, 0.00008]],
            size=len(index),
        )
        benchmark_returns = (
            0.65 * asset_returns[:, 0]
            + 0.20 * asset_returns[:, 1]
            + rng.normal(0.0, 0.004, len(index))
        )
        prices = pd.DataFrame(
            100 * np.cumprod(1 + asset_returns, axis=0),
            index=index,
            columns=["AAA", "BBB"],
        )
        benchmark = pd.Series(
            100 * np.cumprod(1 + benchmark_returns),
            index=index,
            name="BMK",
        )
        volumes = pd.DataFrame(
            {"AAA": 1_000_000.0, "BBB": 800_000.0},
            index=index,
        )

        config = _config()
        result = run_analysis(config, prices, benchmark, volumes)

        self.assertTrue(np.all(result.paths[0] == config.initial_investment))
        self.assertLess(float(np.median(result.terminal_values)), 10_000.0)
        probability_total = sum(
            result.metrics[key]
            for key in (
                "probability_loss",
                "probability_below_target_without_loss",
                "probability_target",
            )
        )
        self.assertAlmostEqual(probability_total, 1.0, places=12)
        self.assertAlmostEqual(
            float(result.risk_contributions["Risk Contribution"].sum()),
            1.0,
            places=10,
        )
        self.assertEqual(result.metrics["margin_call_probability"], 0.0)
        self.assertGreaterEqual(
            result.metrics["es_currency"],
            result.metrics["var_currency"],
        )


if __name__ == "__main__":
    unittest.main()
