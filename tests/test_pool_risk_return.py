import unittest

import pandas as pd

from utils.pool_risk_return_service import is_short_listed, risk_return_point


def _series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


class RiskReturnPointTest(unittest.TestCase):
    def test_cagr_annualizes_by_calendar_days(self) -> None:
        # 365일 뒤 2배 → CAGR 100%.
        close = _series([100.0] + [100.0] * 364 + [200.0])
        point = risk_return_point(close, 12)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point["cagr_pct"], 100.0, places=1)
        self.assertEqual(point["mdd_pct"], 0.0)
        self.assertIsNone(point["calmar"])

    def test_mdd_is_peak_to_trough(self) -> None:
        close = _series([100.0, 120.0, 60.0, 90.0])
        point = risk_return_point(close, 12)
        self.assertEqual(point["mdd_pct"], -50.0)
        self.assertAlmostEqual(point["calmar"], point["cagr_pct"] / 50.0, places=2)

    def test_short_listing_is_dropped_for_a_longer_period(self) -> None:
        close = _series([100.0 + day for day in range(120)])  # 상장 약 4개월
        self.assertTrue(is_short_listed(close, 12))
        self.assertFalse(is_short_listed(close, 3))

    def test_too_few_bars_gives_no_point(self) -> None:
        self.assertIsNone(risk_return_point(_series([100.0]), 12))


if __name__ == "__main__":
    unittest.main()
