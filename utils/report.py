"""
리포트 및 로그 출력을 위한 포맷팅 유틸리티 함수 모음.
"""

from typing import List

def fmt_manwon(v: float) -> str:
    """float 값을 '만원' 단위의 문자열로 포맷합니다."""
    return f"{int(round(v/10_000)):,}만원"

def fmt_signed_pct_1d(x: float) -> str:
    """float 값을 부호가 있는 소수점 1자리 퍼센트 문자열로 포맷합니다."""
    v = round(float(x), 1)
    if v > 0:
        s = f"+{v:.1f}%"
    elif v < 0:
        s = f"{v:.1f}%"
    else:
        s = "-0.0%"
    # 한 자릿수일 때 정렬을 위해 앞에 공백 추가
    if abs(v) < 10.0:
        s = " " + s
    return s

def color_pct_1d(x: float) -> str:
    """float 값을 ANSI 터미널용 색상이 포함된 퍼센트 문자열로 포맷합니다."""
    red = "\x1b[31m"
    blue = "\x1b[34m"
    reset = "\x1b[0m"
    s = fmt_signed_pct_1d(x)
    v = round(float(x), 1)
    return f"{red}{s}{reset}" if v > 0 else f"{blue}{s}{reset}"


def render_table_eaw(
    headers: List[str],
    rows: List[List[str]],
    aligns: List[str],
    amb_wide: bool = True
) -> List[str]:
    """
    리스트의 리스트를 너비를 인식하는 ASCII 테이블 문자열로 렌더링합니다.
    동아시아 문자를 처리합니다.
    """
    from unicodedata import east_asian_width, normalize
    import re

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _clean(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = _ANSI_RE.sub('', s)
        s = normalize('NFKC', s)
        return s

    def _disp_width_eaw(s: str) -> int:
        s = _clean(s)
        w = 0
        for ch in s:
            eaw = east_asian_width(ch)
            if eaw in ('W', 'F') or (amb_wide and eaw == 'A'):
                w += 2
            else:
                w += 1
        return w

    def _pad(s: str, width: int, align: str) -> str:
        s_clean = _clean(s)
        dw = _disp_width_eaw(s_clean)
        if dw >= width:
            return s
        pad = width - dw
        if align == 'right':
            return ' ' * pad + s
        elif align == 'center':
            left = pad // 2
            right = pad - left
            return ' ' * left + s + ' ' * right
        else: # left
            return s + ' ' * pad

    widths = [max(_disp_width_eaw(v) for v in [headers[j]] + [r[j] for r in rows]) for j in range(len(headers))]

    def _hline():
        return '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    out = [_hline()]
    header_cells = [_pad(headers[j], widths[j], 'center' if aligns[j] == 'center' else 'left') for j in range(len(headers))]
    out.append('| ' + ' | '.join(header_cells) + ' |')
    out.append(_hline())
    for r in rows:
        cells = [_pad(r[j], widths[j], aligns[j]) for j in range(len(headers))]
        out.append('| ' + ' | '.join(cells) + ' |')
    out.append(_hline())
    return out