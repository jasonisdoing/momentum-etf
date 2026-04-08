from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

import requests
from pykrx import stock
from pykrx.website.comm import webio

LOGIN_PAGE_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
LOGIN_IFRAME_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
LOGIN_POST_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def load_dotenv() -> None:
    env_path = Path(__file__).resolve().with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key or normalized_key in os.environ:
            continue
        os.environ[normalized_key] = value.strip().strip('"').strip("'")


def install_requests_session(session: requests.Session) -> None:
    # pykrx는 requests.get/post를 직접 호출하므로 동일 세션으로 교체한다.
    webio.requests.get = session.get
    webio.requests.post = session.post


def login_krx() -> requests.Session:
    login_id = str(os.environ.get("KRX_LOGIN_ID") or "").strip()
    login_password = str(os.environ.get("KRX_LOGIN_PASSWORD") or "").strip()
    if not login_id or not login_password:
        raise RuntimeError(
            "GitHub 우회 로그인은 KRX_LOGIN_ID와 KRX_LOGIN_PASSWORD가 필요합니다. "
            "KRX_API_KEY만으로는 data.krx.co.kr 로그인 세션을 만들 수 없습니다."
        )

    session = requests.Session()
    install_requests_session(session)

    session.get(LOGIN_PAGE_URL, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=15)
    session.get(
        LOGIN_IFRAME_URL,
        headers={"User-Agent": DEFAULT_USER_AGENT, "Referer": LOGIN_PAGE_URL},
        timeout=15,
    )

    payload = {
        "mbrNm": "",
        "telNo": "",
        "di": "",
        "certType": "",
        "mbrId": login_id,
        "pw": login_password,
    }
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": LOGIN_PAGE_URL,
        "X-Requested-With": "XMLHttpRequest",
    }

    response = session.post(LOGIN_POST_URL, data=payload, headers=headers, timeout=15)
    data = response.json()
    error_code = str(data.get("_error_code") or "")

    if error_code == "CD011":
        payload["skipDup"] = "Y"
        response = session.post(LOGIN_POST_URL, data=payload, headers=headers, timeout=15)
        data = response.json()
        error_code = str(data.get("_error_code") or "")

    if error_code != "CD001":
        error_message = str(data.get("_error_message") or "알 수 없는 로그인 오류").strip()
        raise RuntimeError(f"KRX 로그인 실패: {error_code} {error_message}".strip())

    return session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pykrx ETF 구성종목(PDF) 조회를 직접 테스트합니다.",
    )
    parser.add_argument("ticker", help="ETF 티커. 예: 117700")
    parser.add_argument(
        "--date",
        dest="date",
        default=None,
        help="조회 일자 YYYYMMDD. 예: 20250407",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="data.krx.co.kr 로그인 세션 우회를 먼저 시도합니다. KRX_LOGIN_ID/KRX_LOGIN_PASSWORD가 필요합니다.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    ticker = str(args.ticker).strip()
    date = str(args.date).strip() if args.date else None

    print(f"ticker={ticker}")
    print(f"date={date or '(생략)'}")
    print(f"login={'사용' if args.login else '미사용'}")
    print("-" * 80)

    try:
        if args.login:
            login_krx()
            print("KRX 로그인 세션 설정 완료")
            print("-" * 80)
        df = stock.get_etf_portfolio_deposit_file(ticker, date)
    except Exception as error:
        print(f"호출 실패: {type(error).__name__}: {error}")
        print("-" * 80)
        traceback.print_exc()
        return 1

    print(f"rows={len(df)}")
    print(f"columns={list(df.columns)}")
    print("-" * 80)

    if df is None or df.empty:
        print("빈 DataFrame 입니다.")
        return 0

    print(df.head(20).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
