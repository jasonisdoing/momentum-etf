"""Streamlit index.html에 Open Graph 메타 태그를 추가하는 스크립트"""

import os
import shutil

import streamlit
from bs4 import BeautifulSoup

# Streamlit index.html 경로 동적 탐색
streamlit_static_dir = os.path.join(os.path.dirname(streamlit.__file__), "static")
streamlit_path = os.path.join(streamlit_static_dir, "index.html")

# 원본 index.html 백업 생성
backup_path = streamlit_path + ".bak"
if not os.path.exists(backup_path):
    shutil.copy2(streamlit_path, backup_path)
    print(f"✅ 백업 생성: {backup_path}")

# 원본 index.html 읽기
with open(streamlit_path, encoding="utf-8") as file:
    html_content = file.read()

# HTML 파싱
soup = BeautifulSoup(html_content, "html.parser")

# 새 메타태그 정의
meta_tags = [
    # 일반 SEO
    {"name": "description", "content": "추세추종 전략 기반 ETF 투자"},
    # Open Graph
    {"property": "og:type", "content": "website"},
    {"property": "og:url", "content": "https://etf.dojason.com/"},
    {"property": "og:title", "content": "Momentum ETF"},
    {"property": "og:description", "content": "추세추종 전략 기반 ETF 투자"},
    {
        "property": "og:image",
        "content": "https://raw.githubusercontent.com/jasonisdoing/momentum-etf/upgrade/static/og-image.png",
    },
    {"property": "og:image:width", "content": "1024"},
    {"property": "og:image:height", "content": "1024"},
    {"property": "og:image:type", "content": "image/png"},
    {"property": "og:site_name", "content": "Momentum ETF"},
    # Twitter Card
    {"name": "twitter:card", "content": "summary_large_image"},
    {"name": "twitter:title", "content": "Momentum ETF"},
    {"name": "twitter:description", "content": "추세추종 전략 기반 ETF 투자"},
    {
        "name": "twitter:image",
        "content": "https://raw.githubusercontent.com/jasonisdoing/momentum-etf/upgrade/static/og-image.png",
    },
]

# head에 새 메타태그 추가
for tag in meta_tags:
    new_tag = soup.new_tag("meta")
    for key, value in tag.items():
        new_tag[key] = value
    soup.head.append(new_tag)

# 수정된 HTML 저장
with open(streamlit_path, "w", encoding="utf-8") as file:
    file.write(str(soup))

print(f"✅ 메타 태그 추가 완료: {streamlit_path}")
print(f"   추가된 메타 태그: {len(meta_tags)}개")
