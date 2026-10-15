"""SPDR 공식 XLSX의 공통 파일 검증과 표 추출."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from xml.etree import ElementTree
from zipfile import ZipFile

_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_spdr_xlsx_rows(content: bytes) -> list[dict[str, str]]:
    """SPDR 공식 XLSX의 첫 시트를 읽고 빈 행 경계를 보존한다."""
    with ZipFile(BytesIO(content)) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(node.text or "" for node in item.findall(".//s:t", _NS)) for item in root]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[dict[str, str]] = []
    previous = 0
    for row in sheet.findall("s:sheetData/s:row", _NS):
        number = int(row.attrib["r"])
        if number > previous + 1:
            rows.append({})
        cells = {}
        for cell in row.findall("s:c", _NS):
            if cell.find("s:f", _NS) is not None:
                raise ValueError("SPDR 공식 파일에 계산되지 않은 수식이 있습니다.")
            kind = cell.attrib.get("t")
            value = cell.findtext("s:v", namespaces=_NS)
            if kind == "s" and value is not None:
                value = strings[int(value)]
            elif kind == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall("s:is/s:t", _NS))
            elif kind == "e":
                raise ValueError("SPDR 공식 파일에 오류 셀이 있습니다.")
            if value is not None and value.strip():
                cells[cell.attrib["r"].rstrip("0123456789")] = value.strip()
        rows.append(cells)
        previous = number
    return rows


def parse_spdr_table_rows(rows: list[dict[str, str]], required: set[str]) -> list[dict[str, str]]:
    header_index = next((index for index, row in enumerate(rows) if required <= set(row.values())), None)
    if header_index is None:
        raise ValueError("SPDR 공식 파일에 필수 컬럼이 없습니다.")
    headers = rows[header_index]
    result = []
    for row in rows[header_index + 1 :]:
        if not row:
            if result:
                break
            continue
        # 상품 명단의 두 번째 헤더에는 필수 컬럼의 값이 없다.
        if not result and not any(row.get(column) for column, title in headers.items() if title in required):
            continue
        result.append({title: row.get(column, "") for column, title in headers.items()})
    if not result:
        raise ValueError("SPDR 공식 파일의 명단이 비었습니다.")
    return result


def parse_spdr_holdings_xlsx(content: bytes, ticker: str, required: set[str]) -> tuple[str, list[dict[str, str]]]:
    """파일의 상품·기준일을 확인하고 전체 구성종목 표를 반환한다."""
    rows = read_spdr_xlsx_rows(content)
    identity = {row["A"]: row.get("B", "") for row in rows if row.get("A") in {"Ticker Symbol:", "Holdings:"}}
    if identity.get("Ticker Symbol:") != ticker:
        raise ValueError(f"SPDR {ticker} 구성종목 파일의 티커가 다릅니다.")
    reference_date = datetime.strptime(identity["Holdings:"], "As of %d-%b-%Y").date().isoformat()
    return reference_date, parse_spdr_table_rows(rows, required)
