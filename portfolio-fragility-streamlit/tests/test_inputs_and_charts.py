"""Regression checks for localized inputs, horizons, drawdowns, and chart metadata."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.analytics import asset_drawdowns
from src.dashboard import fragility_gauge, probability_partition_labels
from src.engine import validate_portfolio
from src.instruments import custom_horizon, parse_localized_number


class InputAndChartTests(unittest.TestCase):
    def test_localized_decimal_weights(self) -> None:
        self.assertEqual(parse_localized_number("20,3"), 20.3)
        self.assertEqual(parse_localized_number("20.3"), 20.3)
        self.assertEqual(parse_localized_number("1.234,5"), 1234.5)
        table = pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT", "GLD"],
                "Weight (%)": ["20,3", "29,7", "50"],
            }
        )
        tickers, weights = validate_portfolio(table)
        self.assertEqual(tickers, ("AAPL", "MSFT", "GLD"))
        np.testing.assert_allclose(weights, (0.203, 0.297, 0.5))

    def test_custom_horizon_uses_daily_resolution(self) -> None:
        sessions, label, calendar_days = custom_horizon(100, "Weeks")
        self.assertEqual(sessions, 483)
        self.assertEqual(label, "100 weeks")
        self.assertEqual(calendar_days, 700)
        self.assertEqual(custom_horizon(1, "Hours")[0], 1)

    def test_asset_drawdown_is_separate_per_ticker(self) -> None:
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        result = SimpleNamespace(
            prices=pd.DataFrame(
                {
                    "AAA": [100, 120, 90, 100, 121],
                    "BBB": [100, 90, 80, 85, 95],
                },
                index=index,
            )
        )
        drawdowns, episodes = asset_drawdowns(result)
        self.assertEqual(list(drawdowns.columns), ["AAA", "BBB"])
        aaa = episodes.set_index("Ticker").loc["AAA"]
        bbb = episodes.set_index("Ticker").loc["BBB"]
        self.assertAlmostEqual(aaa["Maximum Drawdown"], -0.25)
        self.assertAlmostEqual(bbb["Maximum Drawdown"], -0.20)
        self.assertEqual(aaa["Recovery Date"], index[-1])
        self.assertTrue(pd.isna(bbb["Recovery Date"]))

    def test_fragility_gauge_has_no_undefined_title(self) -> None:
        result = SimpleNamespace(metrics={"fragility_score": 42.2})
        payload = json.dumps(fragility_gauge(result).to_plotly_json())
        self.assertNotIn("undefined", payload.lower())

    def test_probability_labels_round_to_exact_partition(self) -> None:
        labels = probability_partition_labels([0.279, 0.195, 0.526])
        values = [float(label.rstrip("%")) for label in labels]
        self.assertAlmostEqual(sum(values), 100.0, places=8)


if __name__ == "__main__":
    unittest.main()
