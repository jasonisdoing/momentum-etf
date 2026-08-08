"""yfinance 섹터·업종의 한글 표기 — 표시 전용 번역 사전(단일 소스).

저장(stock_meta)은 영문 원본을 유지한다 — 업종 상한 그룹핑 키가 미국 풀과
계속 일치해야 하기 때문이다. 화면 응답을 만들 때만 이 사전으로 번역한다.

사전에 없는 값은 **영문 그대로** 반환한다 — 임의 보정 없이, 영문 표시 자체가
"미번역 신규 값" 신호다. 새 값이 보이면 여기에 추가한다.
"""

from __future__ import annotations

# yfinance sector — GICS 유사 11종 고정.
SECTOR_KO: dict[str, str] = {
    "Basic Materials": "소재",
    "Communication Services": "통신서비스",
    "Consumer Cyclical": "경기소비재",
    "Consumer Defensive": "필수소비재",
    "Energy": "에너지",
    "Financial Services": "금융",
    "Healthcare": "헬스케어",
    "Industrials": "산업재",
    "Real Estate": "부동산",
    "Technology": "기술",
    "Utilities": "유틸리티",
}

# yfinance industry — 한국 풀(코스피·코스닥)에 등장하는 값 위주로 관리한다.
INDUSTRY_KO: dict[str, str] = {
    "Advertising Agencies": "광고",
    "Aerospace & Defense": "항공우주·방산",
    "Airlines": "항공사",
    "Apparel Manufacturing": "의류 제조",
    "Asset Management": "자산운용",
    "Auto Manufacturers": "자동차",
    "Auto Parts": "자동차 부품",
    "Banks - Regional": "은행",
    "Biotechnology": "바이오",
    "Building Materials": "건축자재",
    "Capital Markets": "증권",
    "Chemicals": "화학",
    "Communication Equipment": "통신장비",
    "Computer Hardware": "컴퓨터 하드웨어",
    "Confectioners": "제과",
    "Conglomerates": "지주·복합기업",
    "Consumer Electronics": "가전·전자",
    "Credit Services": "여신금융",
    "Department Stores": "백화점",
    "Discount Stores": "할인점",
    "Drug Manufacturers - General": "제약(대형)",
    "Drug Manufacturers - Specialty & Generic": "제약",
    "Electrical Equipment & Parts": "전기장비·부품",
    "Electronic Components": "전자부품",
    "Electronic Gaming & Multimedia": "게임",
    "Electronics & Computer Distribution": "전자기기 유통",
    "Engineering & Construction": "건설",
    "Entertainment": "엔터테인먼트",
    "Farm & Heavy Construction Machinery": "농기계·중장비",
    "Food Distribution": "식품 유통",
    "Furnishings, Fixtures & Appliances": "가구·생활가전",
    "Grocery Stores": "식료품 소매",
    "Health Information Services": "헬스케어 정보서비스",
    "Household & Personal Products": "생활용품·화장품",
    "Information Technology Services": "IT 서비스",
    "Insurance - Diversified": "종합보험",
    "Insurance - Life": "생명보험",
    "Insurance - Property & Casualty": "손해보험",
    "Insurance - Reinsurance": "재보험",
    "Integrated Freight & Logistics": "물류",
    "Internet Content & Information": "인터넷 서비스",
    "Leisure": "레저",
    "Lodging": "호텔·숙박",
    "Marine Shipping": "해운",
    "Medical Care Facilities": "의료기관",
    "Medical Devices": "의료기기",
    "Medical Distribution": "의약품 유통",
    "Metal Fabrication": "금속가공",
    "Oil & Gas Refining & Marketing": "정유",
    "Other Industrial Metals & Mining": "비철금속·광업",
    "Packaged Foods": "음식료",
    "Packaging & Containers": "포장재",
    "Pollution & Treatment Controls": "환경설비",
    "REIT - Office": "리츠(오피스)",
    "Railroads": "철도",
    "Rental & Leasing Services": "임대·리스",
    "Resorts & Casinos": "리조트·카지노",
    "Scientific & Technical Instruments": "정밀계측기기",
    "Security & Protection Services": "보안 서비스",
    "Semiconductor Equipment & Materials": "반도체 장비·소재",
    "Semiconductors": "반도체",
    "Shell Companies": "기업인수목적회사",
    "Software - Application": "소프트웨어(응용)",
    "Software - Infrastructure": "소프트웨어(인프라)",
    "Solar": "태양광",
    "Specialty Chemicals": "정밀화학",
    "Specialty Industrial Machinery": "산업기계",
    "Specialty Retail": "전문 소매",
    "Steel": "철강",
    "Telecom Services": "통신사",
    "Textile Manufacturing": "섬유",
    "Tobacco": "담배",
    "Tools & Accessories": "공구·부품",
    "Utilities - Regulated Electric": "전력",
    "Utilities - Regulated Gas": "도시가스",
    "Utilities - Renewable": "신재생에너지",
}


def sector_ko(value: str | None) -> str:
    """섹터 한글 표기. 사전에 없으면 영문 그대로(미번역 신호)."""
    text = str(value or "").strip()
    return SECTOR_KO.get(text, text)


def industry_ko(value: str | None) -> str:
    """업종 한글 표기. 사전에 없으면 영문 그대로(미번역 신호)."""
    text = str(value or "").strip()
    return INDUSTRY_KO.get(text, text)
