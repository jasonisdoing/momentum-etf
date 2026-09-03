from __future__ import annotations

import unittest

from utils.pool_settings_store import PoolSettingsError, _canonical_benchmark_for_country


class PoolSettingsBenchmarkTest(unittest.TestCase):
    def test_australian_benchmark_gets_asx_prefix(self) -> None:
        result = _canonical_benchmark_for_country(
            {"ticker": "IVV", "name": "잘못 들어온 이름"},
            "au",
            [{"ticker": "ASX:IVV", "name": "iShares S&P 500 ETF"}],
        )

        self.assertEqual(result, {"ticker": "ASX:IVV", "name": "iShares S&P 500 ETF"})

    def test_benchmark_outside_country_is_rejected(self) -> None:
        with self.assertRaisesRegex(PoolSettingsError, "국가\\(au\\)에 등록된 종목이 아닙니다"):
            _canonical_benchmark_for_country(
                {"ticker": "QQQ", "name": "Invesco QQQ Trust"},
                "au",
                [{"ticker": "ASX:IVV", "name": "iShares S&P 500 ETF"}],
            )


if __name__ == "__main__":
    unittest.main()
