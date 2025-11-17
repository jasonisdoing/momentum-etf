from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class MemmapPriceStore:
    """Lazy loader for memmap-backed OHLC data."""

    def __init__(self, catalog: Optional[Dict[str, Any]] = None) -> None:
        catalog = catalog or {}
        self.base_dir = Path(catalog.get("base_dir") or "")
        tickers_meta = catalog.get("tickers") or {}
        if not isinstance(tickers_meta, dict):
            tickers_meta = {}
        self._tickers_meta: Dict[str, Dict[str, Any]] = tickers_meta
        self._frame_cache: Dict[str, pd.DataFrame] = {}

    def get_frame(self, ticker: str) -> Optional[pd.DataFrame]:
        key = str(ticker or "").strip()
        if not key:
            return None
        if key in self._frame_cache:
            return self._frame_cache[key]

        meta = self._tickers_meta.get(key)
        if not meta:
            return None

        index_meta = meta.get("index")
        close_meta = meta.get("close")
        open_meta = meta.get("open")
        if not (index_meta and close_meta and open_meta):
            return None

        try:
            index_arr = np.memmap(
                index_meta["path"],
                mode="r",
                dtype=index_meta.get("dtype", "int64"),
                shape=tuple(index_meta.get("shape") or []),
            )
            close_arr = np.memmap(
                close_meta["path"],
                mode="r",
                dtype=close_meta.get("dtype", "float64"),
                shape=tuple(close_meta.get("shape") or []),
            )
            open_arr = np.memmap(
                open_meta["path"],
                mode="r",
                dtype=open_meta.get("dtype", "float64"),
                shape=tuple(open_meta.get("shape") or []),
            )
            index = pd.to_datetime(np.array(index_arr, copy=True))
            close_series = pd.Series(np.array(close_arr, copy=True), index=index, name=close_meta.get("column", "Close"))
            open_series = pd.Series(np.array(open_arr, copy=True), index=index, name="Open")
            frame = pd.DataFrame({close_series.name: close_series, "Open": open_series})
            self._frame_cache[key] = frame
            return frame
        except Exception:
            return None


def create_memmap_store(catalog: Optional[Dict[str, Any]]) -> Optional[MemmapPriceStore]:
    if not catalog:
        return None
    return MemmapPriceStore(catalog)
