"""Open Graph 메타 태그를 HTML head에 주입하는 유틸리티"""

import streamlit as st
import streamlit.components.v1 as components


def inject_meta_tags():
    """Open Graph 및 Twitter Card 메타 태그를 페이지에 주입합니다."""
    meta_tags = """
    <head>
        <meta property="og:title" content="Momentum ETF" />
        <meta property="og:description" content="추세추종 전략 기반 ETF 투자" />
        <meta property="og:image" content="https://etf.dojason.com/static/og-image.png" />
        <meta property="og:url" content="https://etf.dojason.com/" />
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="Momentum ETF" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Momentum ETF" />
        <meta name="twitter:description" content="추세추종 전략 기반 ETF 투자" />
        <meta name="twitter:image" content="https://etf.dojason.com/static/og-image.png" />
        <meta name="description" content="추세추종 전략 기반 ETF 투자" />
    </head>
    """

    # JavaScript를 사용하여 head에 메타 태그 주입
    components.html(
        f"""
        <script>
        const meta = `{meta_tags}`;
        const parser = new DOMParser();
        const doc = parser.parseFromString(meta, 'text/html');
        const metaTags = doc.head.children;

        for (let tag of metaTags) {{
            const existingTag = document.querySelector(`meta[property="${{tag.getAttribute('property')}}"]`) ||
                               document.querySelector(`meta[name="${{tag.getAttribute('name')}}"]`);
            if (existingTag) {{
                existingTag.remove();
            }}
            document.head.appendChild(tag.cloneNode(true));
        }}
        </script>
        """,
        height=0,
    )
