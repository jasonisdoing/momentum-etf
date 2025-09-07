"""
데이터 조회, 파일 입출력 등 공통으로 사용되는 유틸리티 함수 모음.
"""
import pandas as pd
from typing import Tuple, Optional, List
import logging
from datetime import datetime
import csv
import os


# pykrx가 설치되지 않았을 경우를 대비한 예외 처리
try:
    from pykrx import stock as _stock
except Exception:
    _stock = None

def is_pykrx_available() -> bool:
    """pykrx 모듈이 성공적으로 임포트되었는지 확인합니다."""
    return _stock is not None

def get_today_str() -> str:
    """오늘 날짜를 'YYYYMMDD' 형식의 문자열로 반환합니다."""
    return datetime.now().strftime('%Y%m%d')

def fetch_top_performers(ref_date: pd.Timestamp, count: int) -> List[Tuple[str, str]]:
    """(사용되지 않음) 상위 수익률 주식 종목을 조회합니다."""
    # 이 함수는 현재 사용되지 않지만, 혹시 모를 경우를 위해 남겨둡니다.
    # ETF 버전은 fetch_top_performing_etfs를 사용하세요.
    if not is_pykrx_available():
        logging.getLogger(__name__).error("pykrx가 설치되지 않아 동적 티커 선택을 사용할 수 없습니다.")
        return []
    try:
        # ... (implementation can be left as is or updated)
        return []
    except Exception:
        return []


def fetch_ohlcv(ticker: str, months_back: int = None, months_range: Optional[List[int]] = None, date_range: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """pykrx를 통해 OHLCV 데이터를 조회합니다."""
    if not is_pykrx_available():
        return None

    if date_range and len(date_range) == 2:
        try:
            start_dt_core = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1])
            # Warmup: 코어 시작일보다 4주(28일) 앞서서 데이터를 가져옴
            start_dt = start_dt_core - pd.DateOffset(days=28)
        except (ValueError, TypeError):
            logging.getLogger(__name__).error(f"잘못된 date_range 형식: {date_range}. 'YYYY-MM-DD' 형식을 사용해야 합니다.")
            return None
    else:
        now = pd.to_datetime(get_today_str())
        if months_range is not None and len(months_range) == 2:
            start_off, end_off = months_range
            start_dt_core = now - pd.DateOffset(months=int(start_off))
            end_dt = now - pd.DateOffset(months=int(end_off))
            # Warmup: 코어 시작일보다 4주(28일) 앞서서 데이터를 가져옴
            start_dt = start_dt_core - pd.DateOffset(days=28)
        else:
            if months_back is None:
                months_back = 12
            start_dt = now - pd.DateOffset(months=int(months_back))
            end_dt = now
    
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt
    start = start_dt.strftime('%Y%m%d')
    end = end_dt.strftime('%Y%m%d')
    try:
        df = _stock.get_market_ohlcv_by_date(start, end, ticker)
        if df is None or len(df) == 0:
            return None
        return df.rename(columns={
            '시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close', '거래량': 'Volume'
        })
    except Exception as e:
        logging.getLogger(__name__).exception(f'{ticker} 조회 실패: {e}')
        return None


def read_tickers_file(path: str = 'tickers.txt') -> List[Tuple[str, str]]:
    """tickers.txt 파일에서 (티커, 이름) 목록을 읽어옵니다."""
    items: List[Tuple[str, str]] = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                parts = [p.strip() for p in s.replace('\t', ',').split(',') if p.strip()]
                if len(parts) == 1:
                    parts = s.split()
                if len(parts) == 1:
                    ticker, name = parts[0], ''
                else:
                    ticker, name = parts[0], ' '.join(parts[1:])
                items.append((ticker, name))
    except FileNotFoundError:
        logging.getLogger(__name__).error(f'{path} 파일을 찾을 수 없습니다.')
    return items


def read_holdings_file(path: str = 'data/holdings.csv') -> dict:
    """holdings.csv 파일에서 보유 현황을 읽어옵니다."""
    res = {}
    if not os.path.exists(path):
        return res
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                tkr = (row[0] or '').strip()
                if not tkr or tkr.startswith('#'):
                    continue
                
                def _to_float(x):
                    try:
                        s = str(x).replace(',', '').strip()
                        return float(s) if s != '' else None
                    except Exception: return None
                def _to_int(x):
                    try:
                        v = _to_float(x)
                        return int(v) if v is not None else None
                    except Exception: return None

                shares = _to_int(row[2] if len(row) > 2 else None) # New: ticker,name,shares,amount
                avg_cost = _to_float(row[3] if len(row) > 3 else None)
                if shares is None or shares <= 0:
                    shares = _to_int(row[1] if len(row) > 1 else None) # Old: ticker,shares,avg_cost
                    avg_cost = _to_float(row[2] if len(row) > 2 else None)

                if shares is not None and shares > 0:
                    res[tkr] = {"shares": int(shares), "avg_cost": avg_cost}
        if res:
            return res
    except Exception:
        pass
    # Fallback for legacy TSV/whitespace format
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'): continue
            parts = [p.strip() for p in s.replace('\t', ',').split(',') if p.strip()]
            if len(parts) < 2: parts = s.split()
            if len(parts) < 2: continue
            tkr = parts[0]
            try:
                sh = int(float(parts[1]))
            except Exception: continue
            ac = None
            if len(parts) >= 3:
                try:
                    ac = float(parts[2])
                except Exception: ac = None
            res[tkr] = {"shares": int(sh), "avg_cost": ac}
    return res