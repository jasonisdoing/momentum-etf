"""
리포트 및 로그 출력을 위한 포맷팅 유틸리티 함수 모음.
"""

import io
import re
from typing import List
from unicodedata import east_asian_width, normalize

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def format_kr_money(value: float) -> str:
    """금액을 '억', '만' 단위를 포함한 한글 문자열로 포맷합니다."""
    if value is None or not isinstance(value, (int, float)):
        return "-"
    val_int = int(round(value))
    if val_int == 0:
        return "0원"

    sign = "-" if val_int < 0 else ""
    val_abs = abs(val_int)

    eok = val_abs // 100000000
    man = (val_abs % 100000000) // 10000

    parts = []
    if eok > 0:
        parts.append(f"{eok:,}억")
    if man > 0:
        parts.append(f"{man:,}만")

    if not parts:
        # 억, 만 단위가 없는 작은 금액
        return f"{sign}{val_abs:,}원"

    return sign + " ".join(parts) + "원"


def format_aud_money(value: float) -> str:
    """금액을 호주 달러(A$) 형식의 문자열로 포맷합니다."""
    if value is None:
        return "-"
    return f"{value:,.2f}"


def format_aud_price(value: float) -> str:
    """호주 주식 가격을 소수점 4자리까지 포맷합니다."""
    if value is None:
        return "-"
    return f"{value:,.4f}"


def render_table_eaw(headers: List[str], rows: List[List[str]], aligns: List[str]) -> List[str]:
    """
    동아시아 문자 너비를 고려하여 리스트 데이터를 ASCII 테이블 문자열로 렌더링합니다.
    """

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _clean(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = _ANSI_RE.sub("", s)
        s = normalize("NFKC", s)
        return s

    def _disp_width_eaw(s: str) -> int:
        """동아시아 문자를 포함한 문자열의 실제 터미널 출력 너비를 계산합니다."""
        s = _clean(s)
        w = 0
        for ch in s:
            # 박스 드로잉 문자는 터미널에서 넓게 렌더링되는 경우가 많습니다.
            if "\u2500" <= ch <= "\u257f":
                w += 2
                continue
            eaw = east_asian_width(ch)
            # 'Ambiguous'(A) 문자를 Wide로 처리하여 대부분의 터미널에서 정렬이 깨지지 않도록 합니다.
            if eaw in ("W", "F", "A"):
                w += 2
            else:
                w += 1
        return w

    def _pad(s: str, width: int, align: str) -> str:
        """주어진 너비와 정렬에 맞게 문자열에 패딩을 추가합니다."""
        s_str = str(s)
        s_clean = _clean(s_str)
        dw = _disp_width_eaw(s_clean)
        if dw >= width:
            return s_str
        pad = width - dw
        if align == "right":
            return " " * pad + s_str
        elif align == "center":
            left = pad // 2
            right = pad - left
            return " " * left + s_str + " " * right
        else:  # 왼쪽 정렬
            return s_str + " " * pad

    widths = [
        max(_disp_width_eaw(v) for v in [headers[j]] + [r[j] for r in rows])
        for j in range(len(headers))
    ]

    def _hline():
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    out = [_hline()]
    header_cells = [
        _pad(headers[j], widths[j], "center" if aligns[j] == "center" else "left")
        for j in range(len(headers))
    ]
    out.append("| " + " | ".join(header_cells) + " |")
    out.append(_hline())
    for r in rows:
        cells = [_pad(r[j], widths[j], aligns[j]) for j in range(len(headers))]
        out.append("| " + " | ".join(cells) + " |")
    out.append(_hline())
    return out


def render_table_html(headers: List[str], rows: List[List[str]], aligns: List[str]) -> str:
    """리스트 데이터를 HTML 테이블 문자열로 렌더링합니다. (행 높이 여유 있게 표시)"""
    html = '<table border="1" style="border-collapse: collapse; width: 100%;">'
    # Header
    html += "<thead><tr>"
    for i, h in enumerate(headers):
        align_style = f"text-align: {aligns[i]};" if i < len(aligns) else ""
        html += f'<th style="padding: 8px; {align_style} line-height: 2;">{h}</th>'
    html += "</tr></thead>"
    # Body
    html += "<tbody>"
    for r in rows:
        html += "<tr>"
        for i, cell in enumerate(r):
            align_style = f"text-align: {aligns[i]};" if i < len(aligns) else ""
            html += f'<td style="padding: 8px; {align_style} line-height: 2;">{cell}</td>'
        html += "</tr>"
    html += "</tbody></table>"
    return html


def render_table_as_image(
    headers: List[str], rows: List[List[str]], aligns: List[str]
) -> io.BytesIO:
    """Render list data as a table image using matplotlib."""
    matplotlib.use("Agg")

    try:
        plt.rc("font", family="AppleGothic")
        # Windows라면 "Malgun Gothic"
    except Exception as e:
        print(f"Warning: Could not set Korean font. Error: {e}")

    df = pd.DataFrame(rows, columns=headers)

    # --- Figure out table dimensions ---
    fig, ax = plt.subplots(figsize=(10, 3))  # 조금 넉넉한 사이즈
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )

    # --- Style the table ---
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # 모든 셀 높이를 고정 (예: 0.08 → 값은 상황에 따라 조절)
    fixed_height = 0.08
    for (i, j), cell in table.get_celld().items():
        cell.set_height(fixed_height)
        cell.set_edgecolor("lightgrey")

        # Header 스타일
        if i == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4C5A65")

        # 데이터 행 배경색
        if i > 0:
            status = df.iloc[i - 1, 2]  # 3번째 컬럼 기준 예시
            if status == "HOLD":
                cell.set_facecolor("#E6F5FF")
            elif status == "SOLD":
                cell.set_facecolor("#F0F0F0")

        # 정렬
        if aligns and j < len(aligns):
            cell.set_text_props(ha=aligns[j])

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf
