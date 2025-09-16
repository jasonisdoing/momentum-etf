#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI를 사용하여 ETF를 분류하는 유틸리티 함수.
"""

try:
    import google.generativeai as genai
except ImportError:
    genai = None

SECTORS = [
    "AI",
    "소프트웨어",
    "바이오",
    "배터리",
    "리튬",
    "금융",
    "반도체",
    "자동차",
    "산업인프라",
    "소비재",
    "건설",
    "조선",
    "방산",
    "양자",
    "친환경",
    "로봇",
    "에너지",
    "고배당",
    "테크(일반)",
    "금",
    "자원/광산",
    "리츠",
    "채권",
    "글로벌(지수)",
    "미국(지수)",
    "한국(지수)",
    "호주(지수)",
    "중국(지수)",
    "일본(지수)",
    "유럽(지수)",
    "Other",
]


def classify_etf_with_ai(etf_name: str, model) -> str:
    """Gemini AI를 사용하여 ETF 이름을 분석하고 가장 적합한 섹터를 반환합니다."""
    if genai is None:
        raise ImportError(
            "google.generativeai is not installed. Please run 'pip install google-generativeai'"
        )

    prompt = f"""
    ETF 이름 "{etf_name}"을(를) 다음 카테고리 중 가장 적합한 하나로 분류해줘.
    카테고리 목록: {', '.join(SECTORS)}

    규칙:
    1. 산업/테마
       - 'AI' → "AI"
       - 'Software', '소프트웨어', '인터넷' → "소프트웨어"
       - 'Bio', '바이오', 'Healthcare', '헬스케어', '제약' → "바이오"
       - 'Battery', '배터리' → "배터리"
       - 'Lithium', '리튬' → "리튬"
       - '금융', '은행', 'Financials', 'Bank', '보험', '증권' → "금융"
       - 'Semiconductor', '반도체' → "반도체"
       - '자동차', 'Automotive' → "자동차"
       - '중공업', '기계장비', '운송', '산업재' -> "산업인프라"
       - '소비재' -> "소비재"
       - '건설' -> "건설"
       - '조선', '해운' -> "조선"
       - '방산' -> "방산"
       - '양자' -> "양자"
       - 'Sustainability', '친환경', '에코', '클린' → "친환경"
       - 'Robot', '로봇' → "로봇"
       - 'Energy', '에너지' → "에너지"
       - '고배당', 'Yield' → "고배당"
       - 'Technology' -> "테크(일반)"

    2. 원자재/리소스
       - 'Gold', '금' → "금"
       - '자원', 'Resource', 'Mining', '광산', '석유', '철강' → "자원/광산"

    3. 자산 유형
       - '리츠', 'REITs', '부동산', 'Property', 'Real Estate' → "리츠"
       - 'Bond', '채권', 'Fixed Income', 'Credit' → "채권"

    4. 지수형
       - 'Global', 'World', 'International', 'MSCI', '글로벌' → "글로벌(지수)"
       - 'S&P', 'Nasdaq', 'Dow', '미국', 'USA', 'US' → "미국(지수)"
       - 'Korea', 'KOR', 'KOSPI', '코스피', '코스닥', 'Kodex', 'TIGER', 'ACE', 'HANARO' 등 한국 브랜드 → "한국(지수)"
       - 'Australia', 'ASX', '호주' → "호주(지수)"
       - 'China', '차이나' → "중국(지수)"
       - 'Japan', '재팬', '일본' → "일본(지수)"
       - 'Europe', '유럽' → "유럽(지수)"
       단, 그 국가의 어떤 섹터에 포함되어 있으면 섹터로 분류한다

    5. 위 조건에 해당하지 않으면 → "Other"

    오직 카테고리 이름 하나만 응답해줘.
    """
    response = model.generate_content(prompt)
    category = response.text.strip()
    return category if category in SECTORS else "Other"
