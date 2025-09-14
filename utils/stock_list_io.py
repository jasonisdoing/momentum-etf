import os
import json
from typing import List, Dict


def _get_data_dir():
    """Helper to get the absolute path to the 'data' directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "data")


def get_stocks(country: str) -> List[Dict[str, str]]:
    """
    'data/' 디렉토리의 JSON 파일에서 종목 목록을 로드합니다.
    - 주식: {country}_stock.json
    - ETF: {country}_etf.json
    - 파일 형식: [{"category": "...", "tickers": [{"ticker": "...", "name": "..."}]}]
    """
    all_stocks = []
    seen_tickers = set()
    data_dir = _get_data_dir()

    for stock_type in ["stock", "etf"]:
        file_path = os.path.join(data_dir, f"{country}_{stock_type}.json")
        if not os.path.exists(file_path):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"경고: '{file_path}' 파일의 형식이 리스트가 아닙니다. 건너뜁니다.")
                    continue

                for category_block in data:
                    if not isinstance(category_block, dict) or 'tickers' not in category_block:
                        continue

                    category_name = category_block.get('category', 'Uncategorized')
                    tickers_list = category_block.get('tickers', [])
                    if not isinstance(tickers_list, list):
                        continue

                    for item in tickers_list:
                        if not isinstance(item, dict) or not item.get('ticker'):
                            continue

                        ticker = item['ticker']
                        if ticker in seen_tickers:
                            continue

                        seen_tickers.add(ticker)
                        if 'type' not in item or not item['type']:
                            item['type'] = stock_type
                        item['category'] = category_name
                        all_stocks.append(item)
        except json.JSONDecodeError as e:
            print(f"오류: '{file_path}' JSON 파일 파싱 실패 - {e}")
        except Exception as e:
            print(f"경고: '{file_path}' 파일 읽기 실패 - {e}")

    return all_stocks
