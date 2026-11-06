"""Global X 공식 페이지의 공개 JSON 데이터 판독."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from bs4 import BeautifulSoup


def read_globalx_page_records(soup: BeautifulSoup) -> dict[str, Any]:
    """공식 페이지가 전달한 다운로드 설정을 JSON으로 읽는다."""
    chunks = []
    for script in soup.select("script"):
        match = re.fullmatch(r"self\.__next_f\.push\((.*)\)", script.get_text(), re.DOTALL)
        if match:
            entry = json.loads(match[1])
            if entry[0] == 1:
                chunks.append(entry[1])
    records = {}
    for line in "".join(chunks).splitlines():
        match = re.fullmatch(r"([a-f0-9]+):([\[{].*)", line)
        if match:
            records[match[1]] = json.loads(match[2])
    return records


def iter_globalx_page_props(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_globalx_page_props(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_globalx_page_props(child)
