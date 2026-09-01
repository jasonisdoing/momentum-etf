"""
리포트 및 로그 출력을 위한 포맷팅 유틸리티 함수 모음.
"""


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
