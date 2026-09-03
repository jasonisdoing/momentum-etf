from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from scripts.stock_price_cache_updater import _backfill_missing_volumes_from_toss, _notify_cache_issues
from utils.new_high_service import build_price_panel
from utils.trade_value import latest_trade_value_fields, trade_value_multiplier_series


class TradeValueTest(unittest.TestCase):
    def test_multiplier_uses_latest_twenty_valid_observations(self) -> None:
        index = pd.date_range("2026-07-01", periods=21, freq="D")
        values = pd.Series([float(value) for value in range(1, 22)], index=index)
        values.iloc[5] = float("nan")

        result = trade_value_multiplier_series(values)
        expected_window = values.dropna().iloc[-20:]

        self.assertAlmostEqual(result.iloc[-1], expected_window.iloc[-1] / expected_window.mean())
        self.assertTrue(pd.isna(result.iloc[5]))

    def test_batch_fields_use_same_window_definition(self) -> None:
        index = pd.date_range("2026-07-01", periods=21, freq="D")
        close = pd.Series(10.0, index=index)
        volume = pd.Series([float(value) for value in range(1, 22)], index=index)
        volume.iloc[5] = float("nan")

        fields = latest_trade_value_fields(close, volume)
        self.assertIsNotNone(fields)
        assert fields is not None
        values = (close * volume).dropna().iloc[-20:]
        self.assertAlmostEqual(fields["trade_value_mult"], round(values.iloc[-1] / values.mean(), 4))
        self.assertEqual(fields["trade_value_sum19"], round(float(values.iloc[1:].sum()), 2))

    def test_new_high_panel_uses_unadjusted_close_for_trade_value(self) -> None:
        index = pd.date_range("2026-08-27", periods=2, freq="D")
        frame = pd.DataFrame(
            {
                "Open": [90.0, 91.0],
                "High": [101.0, 102.0],
                "Close": [95.0, 96.0],
                "unadjusted_close": [100.0, 101.0],
                "Volume": [10.0, 20.0],
            },
            index=index,
        )

        panel = build_price_panel([{"ticker": "AMGN", "pool": "us_stock"}], {"AMGN": frame})

        self.assertEqual(panel["value"].at[index[-1], "AMGN"], 2020.0)


class TossVolumeBackfillTest(unittest.TestCase):
    @patch("services.toss_market_service.fetch_toss_us_daily_ohlcv")
    @patch("utils.realtime_quotes.resolve_toss_us_product_codes")
    @patch("utils.trading_calendar.is_market_day_completed", return_value=True)
    @patch("utils.trading_calendar.get_trading_days")
    @patch("utils.cache_utils.save_cached_frame")
    @patch("utils.cache_utils.load_cached_frame")
    def test_backfill_changes_only_missing_volume(
        self,
        load_cached_frame,
        save_cached_frame,
        get_trading_days,
        _is_market_day_completed,
        resolve_product_codes,
        fetch_daily,
    ) -> None:
        missing_day = pd.Timestamp.now().normalize() - pd.Timedelta(days=3)
        prior_day = missing_day - pd.Timedelta(days=1)
        cached = pd.DataFrame(
            {
                "Open": [99.0, 100.0],
                "High": [101.0, 102.0],
                "Low": [98.0, 99.0],
                "Close": [100.0, 101.0],
                "Volume": [1000.0, float("nan")],
            },
            index=[prior_day, missing_day],
        )
        load_cached_frame.return_value = cached
        get_trading_days.return_value = [missing_day]
        resolve_product_codes.return_value = {"AMGN": "AMGN"}
        fetch_daily.return_value = [{"date": str(missing_day.date()), "volume": 1739954.0}]

        report = _backfill_missing_volumes_from_toss("us_stock", ["AMGN"])

        self.assertEqual(report["filled_tickers"], 1)
        self.assertEqual(report["filled_days"], {str(missing_day.date()): 1})
        saved = save_cached_frame.call_args.args[2]
        self.assertEqual(saved.at[missing_day, "Volume"], 1739954.0)
        pd.testing.assert_series_equal(
            saved.loc[missing_day, ["Open", "High", "Low", "Close"]],
            cached.loc[missing_day, ["Open", "High", "Low", "Close"]],
        )


class CacheIssueNotificationTest(unittest.TestCase):
    @patch("utils.notification.send_slack_message_v2")
    @patch("scripts.stock_price_cache_updater._recent_trading_days", return_value={"2026-09-03"})
    def test_old_suspicious_date_is_logged_without_slack(self, _recent_days, send_slack) -> None:
        _notify_cache_issues(
            [
                {
                    "pool": "aus_etf",
                    "country_code": "au",
                    "purged_dates": ["2025-10-24"],
                }
            ],
            full_refresh=True,
        )

        send_slack.assert_not_called()


if __name__ == "__main__":
    unittest.main()
