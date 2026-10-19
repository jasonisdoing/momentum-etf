"""SPDR 공식 XLSX의 공통 파일 검증과 표 추출."""

from __future__ import annotations

from datetime import datetime

from services.xlsx_holdings_service import parse_xlsx_table_rows, read_xlsx_rows


def parse_spdr_holdings_xlsx(content: bytes, ticker: str, required: set[str]) -> tuple[str, list[dict[str, str]]]:
    """파일의 상품·기준일을 확인하고 전체 구성종목 표를 반환한다."""
    rows = read_xlsx_rows(content)
    identity = {row["A"]: row.get("B", "") for row in rows if row.get("A") in {"Ticker Symbol:", "Holdings:"}}
    if identity.get("Ticker Symbol:") != ticker:
        raise ValueError(f"SPDR {ticker} 구성종목 파일의 티커가 다릅니다.")
    reference_date = datetime.strptime(identity["Holdings:"], "As of %d-%b-%Y").date().isoformat()
    return reference_date, parse_xlsx_table_rows(rows, required)
