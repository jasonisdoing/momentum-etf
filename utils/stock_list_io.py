import json
import os
from typing import Dict, List


def _get_data_dir():
    """Helper to get the absolute path to the 'data' directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "data")


def get_etfs(country: str) -> List[Dict[str, str]]:
    """
    'data/stocks/{country}.json' 파일에서 종목 목록을 반환합니다.
    """
    all_etfs = []
    seen_tickers = set()

    file_path = os.path.join(_get_data_dir(), "stocks", f"{country}.json")
    if not os.path.exists(file_path):
        return all_etfs

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"경고: '{file_path}' 파일의 형식이 리스트가 아닙니다. 건너뜁니다.")
                return all_etfs

            for category_block in data:
                if not isinstance(category_block, dict) or "tickers" not in category_block:
                    continue

                category_name = category_block.get("category", "Uncategorized")
                tickers_list = category_block.get("tickers", [])
                if not isinstance(tickers_list, list):
                    continue

                for item in tickers_list:
                    if not isinstance(item, dict) or not item.get("ticker"):
                        continue

                    ticker = item["ticker"]
                    ticker_norm = str(ticker).strip()
                    if not ticker_norm or ticker_norm in seen_tickers:
                        continue

                    seen_tickers.add(ticker_norm)

                    new_item = dict(item)
                    new_item["ticker"] = ticker_norm
                    new_item["type"] = "etf"
                    new_item["category"] = category_name
                    new_item["recommend_enabled"] = not (item.get("recommend_enabled") is False)
                    all_etfs.append(new_item)
    except json.JSONDecodeError as e:
        print(f"오류: '{file_path}' JSON 파일 파싱 실패 - {e}")
    except Exception as e:
        print(f"경고: '{file_path}' 파일 읽기 실패 - {e}")

    return all_etfs


def save_etfs(country: str, data: List[Dict]):
    """
    주어진 데이터를 'data/stocks/{country}.json' 파일에 저장합니다.
    """
    stocks_data_dir = os.path.join(_get_data_dir(), "stocks")
    os.makedirs(stocks_data_dir, exist_ok=True)
    file_path = os.path.join(stocks_data_dir, f"{country}.json")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"성공: {len(data)}개 카테고리의 종목 정보가 '{file_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"오류: '{file_path}' 파일 저장 실패 - {e}")
        raise


def get_etf_categories(country: str) -> List[str]:
    """
    지정된 국가의 모든 ETF 카테고리 목록을 반환합니다.
    """
    categories = set()

    file_path = os.path.join(_get_data_dir(), "stocks", f"{country}.json")
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for category_block in data:
                    if isinstance(category_block, dict) and "category" in category_block:
                        categories.add(category_block["category"])
    except Exception as e:
        print(f"경고: '{file_path}' 파일에서 카테고리 읽기 실패 - {e}")

    return sorted(list(categories))
