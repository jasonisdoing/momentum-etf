"""KRX 정보데이터시스템(data.krx.co.kr) 로그인 헬퍼 — 현재 사용처 없음, 보관용.

KRX 가 2026-02-27 부터 정보데이터시스템 조회에 로그인(Data Marketplace 계정)을
요구한다. 이 모듈은 로그인 세션을 만들어 pykrx 내부(webio)에 주입한다
(pykrx 이슈 #276 방식). 나중에 KRX 데이터가 다시 필요해지면 이렇게 쓴다:

    from services.krx_data_login import login
    login()                      # .env 의 KRX_DATA_ID / KRX_DATA_PW 사용
    from pykrx import stock
    df = stock.get_market_ohlcv_by_ticker("20260819", market="ALL")
    # 투자자별 순매수: stock.get_market_net_purchases_of_equities(f, t, "ALL", "사모")
    # 펀더멘털(PER·DPS 등): stock.get_market_fundamental_by_ticker(date, market="ALL")

이력·주의사항 (2026-08 수급 추종 전략 실험에서 확인):
  - **비공식 우회**다. KRX 약관(제10조 제2호)은 자동화 수단의 무단 수집을 금지하며,
    실제로 대량 조회(대기 없이 수천 콜)가 감지되어 IP 가 1일 차단된 이력이 있다.
    쓰게 되면 소량·저빈도로만, 콜 간 0.2초 이상 간격을 권장한다.
  - 로그인하면 브라우저에 로그인돼 있던 기존 세션이 끊길 수 있다(skipDup 처리).
  - 대량·정기 수집이 필요하면 공식 경로(화면 다운로드, 데이터 상품, Open API)를 쓸 것.
  - 당시 수집물: 투자자별 일별 순매수(투신·사모 등 12주체)·전 종목 일별 시세/시총 2년치.
    MongoDB `investor_flow_daily` / `krx_market_daily` 에 쌓았다가 전략 폐기와 함께 삭제.
"""

from __future__ import annotations

import os

import requests

from utils.logger import get_app_logger

logger = get_app_logger()

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
_session = requests.Session()
_logged_in = False


def _inject_session() -> None:
    """pykrx 의 모든 호출이 로그인 쿠키가 실린 공유 세션을 쓰게 한다."""
    from pykrx.website.comm import webio

    def _post_read(self, **params):
        return _session.post(self.url, headers=self.headers, data=params)

    def _get_read(self, **params):
        return _session.get(self.url, headers=self.headers, params=params)

    webio.Post.read = _post_read
    webio.Get.read = _get_read


def login() -> None:
    """KRX Data Marketplace 로그인 — 세션 쿠키 확보 후 pykrx 에 주입. 실패는 명시적 에러."""
    global _logged_in
    if _logged_in:
        return
    login_id, login_pw = os.environ.get("KRX_DATA_ID"), os.environ.get("KRX_DATA_PW")
    if not login_id or not login_pw:
        raise RuntimeError(".env 에 KRX_DATA_ID / KRX_DATA_PW 가 필요합니다 (Data Marketplace 계정).")

    login_page = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
    login_jsp = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
    login_url = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"

    _session.get(login_page, headers={"User-Agent": _UA}, timeout=15)
    _session.get(login_jsp, headers={"User-Agent": _UA, "Referer": login_page}, timeout=15)
    payload = {"mbrNm": "", "telNo": "", "di": "", "certType": "", "mbrId": login_id, "pw": login_pw}
    headers = {"User-Agent": _UA, "Referer": login_page}

    def _post_login() -> dict:
        response = _session.post(login_url, data=payload, headers=headers, timeout=15)
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"KRX 로그인 응답이 JSON 이 아닙니다 (HTTP {response.status_code}) — "
                "일시 차단·점검일 수 있으니 몇 분 뒤 다시 실행하세요."
            ) from exc

    data = _post_login()
    if data.get("_error_code") == "CD011":  # 중복 로그인 — 기존 세션(브라우저)을 끊고 진행
        payload["skipDup"] = "Y"
        data = _post_login()
    if data.get("_error_code") != "CD001":
        raise RuntimeError(f"KRX 로그인 실패 (code={data.get('_error_code')})")
    _inject_session()
    _logged_in = True
    logger.info("[KRX-LOGIN] 로그인 성공")
