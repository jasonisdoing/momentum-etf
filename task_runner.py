import logging
import os

from flask import Flask, jsonify, request

# 로거 설정
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# status.py의 main 함수를 임포트합니다.
# 이 파일이 실행될 때 필요한 환경변수(DB 연결 등)가 설정되어 있어야 합니다.
try:
    from scripts.snapshot_bithumb_balances import main as snapshot_equity_main
    from scripts.sync_bithumb_accounts_to_trades import main as sync_trades_main
    from status import main as run_status_main

    IMPORTS_OK = True
except ImportError as e:
    logging.error(f"필수 모듈 임포트 실패: {e}")
    IMPORTS_OK = False

app = Flask(__name__)

# Cloud Scheduler에서 호출할 때 확인할 인증 토큰 (환경 변수에서 읽어옴)
AUTH_TOKEN = os.environ.get("SCHEDULER_AUTH_TOKEN")
if not AUTH_TOKEN:
    logging.warning("SCHEDULER_AUTH_TOKEN 환경 변수가 설정되지 않았습니다. 보안에 유의하세요.")


@app.before_request
def check_auth():
    # 로컬 개발 환경 등에서 토큰 없이 테스트할 수 있도록 허용
    if not AUTH_TOKEN:
        return

    auth_header = request.headers.get("Authorization")
    expected_header = f"Bearer {AUTH_TOKEN}"
    if auth_header != expected_header:
        logging.warning(f"잘못된 인증 시도: {auth_header}")
        return jsonify({"error": "Unauthorized"}), 401


@app.route("/run-task", methods=["POST"])
def handle_task():
    if not IMPORTS_OK:
        return jsonify({"error": "Server is not configured correctly, missing modules."}), 500

    data = request.get_json()
    if not data or "country" not in data:
        return jsonify({"error": "Missing 'country' in request body"}), 400

    country = data["country"]
    logging.info(f"'{country}' 국가에 대한 작업 요청을 받았습니다.")

    try:
        if country == "coin":
            sync_trades_main()
            snapshot_equity_main()
        run_status_main(country=country)
        logging.info(f"'{country}' 국가 작업이 성공적으로 완료되었습니다.")
        return jsonify({"status": "success", "country": country}), 200
    except Exception as e:
        logging.exception(f"'{country}' 국가 작업 실행 중 오류 발생")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Google Cloud Run은 PORT 환경 변수를 사용하여 컨테이너가 수신 대기할 포트를 알려줍니다.
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
