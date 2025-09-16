import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, Dict, List

import requests

from utils.env import load_env_if_present


class BithumbV2Client:
    """Minimal client for Bithumb API v2 to read accounts (balances + avg buy price).

    Auth: JWT Bearer with payload {access_key, nonce(uuid4), timestamp(ms)} signed with secret.
    Endpoint: GET /v1/accounts (per Bithumb v2 docs), returns list of assets.
    Expected fields per asset (defensive parsing):
      - currency (e.g., 'BTC', 'ETH', 'KRW')
      - total_balance or balance/quantity
      - avg_buy_price or avg_buy_price_krw
    """

    BASE_URL = "https://api.bithumb.com"

    def __init__(self):
        # Ensure .env is loaded for web app runs (Streamlit) and CLI alike
        try:
            load_env_if_present()
        except Exception:
            pass
        self.access_key = os.environ.get("BITHUMB_V2_API_KEY") or os.environ.get("BITHUMB_API_KEY")
        self.secret_key = os.environ.get("BITHUMB_V2_API_SECRET") or os.environ.get(
            "BITHUMB_API_SECRET"
        )
        if not self.access_key or not self.secret_key:
            raise RuntimeError(
                "BITHUMB_V2_API_KEY / BITHUMB_V2_API_SECRET (or v1 vars) are required"
            )
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "MomentumEtf/1.0"})

    def _auth_headers(self) -> Dict[str, str]:
        payload = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(round(time.time() * 1000)),
        }

        # Minimal JWT HS256 (header.payload.signature), base64url without padding
        def b64url(b: bytes) -> bytes:
            return base64.urlsafe_b64encode(b).rstrip(b"=")

        header = {"alg": "HS256", "typ": "JWT"}
        signing_input = b".".join(
            [
                b64url(json.dumps(header, separators=(",", ":")).encode("utf-8")),
                b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
            ]
        )
        sig = hmac.new(self.secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
        token = b".".join([signing_input, b64url(sig)]).decode("utf-8")
        return {"Authorization": f"Bearer {token}"}

    def accounts(self) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/v1/accounts"
        r = self.session.get(url, headers=self._auth_headers(), timeout=10)
        r.raise_for_status()
        data = r.json()
        # Expected shape: {status:'0000', data:[{...}, ...]}
        if isinstance(data, dict) and data.get("data"):
            items = data["data"]
            if isinstance(items, list):
                return items
        # Some docs show direct list
        if isinstance(data, list):
            return data
        return []
