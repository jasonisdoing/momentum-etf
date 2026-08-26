"""DB 컬렉션 카탈로그 — 어떤 테이블이 무엇에 속하는지의 **단일 소스**.

`/data-tables` 화면이 이 목록을 그대로 보여준다. 화면이 목록을 따로 들지 않는 이유는,
같은 목록을 **종목풀·계좌 삭제**와 **고아 데이터 점검**도 써야 하기 때문이다. 예전에는
"무엇을 지워야 하는가" 가 `delete_pool`/`delete_account` 안에 흩어져 있어서, 컬렉션이
하나 늘 때마다 조용히 빠졌다(그래서 `previous_stock_cache_meta` 에 삭제된 풀의 문서가
수백 건 남았다). 여기 한 줄을 더하면 화면·삭제·점검이 함께 따라온다.

**카탈로그에 없는 컬렉션은 화면에 '미분류'로 뜬다.** 새 컬렉션을 만들었는데 여기 등록하지
않으면 바로 눈에 띄라고 일부러 그렇게 둔다 — 조용히 넘어가지 않는 것이 목적이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from utils.logger import get_app_logger

logger = get_app_logger()

# 분류 — 화면 그룹이자 삭제 정책의 기준.
Category = Literal["pool", "account", "aggregate", "config", "reference", "runtime", "personal"]

CATEGORY_LABELS: dict[str, str] = {
    "pool": "종목풀",
    "account": "계좌",
    "aggregate": "집계·이력",
    "config": "설정",
    "reference": "참조",
    "runtime": "실행 상태",
    "personal": "개인",
}

# 분류별 삭제 정책 — 화면이 그대로 보여준다.
CATEGORY_POLICIES: dict[str, str] = {
    "pool": "종목풀 삭제 시 함께 삭제",
    "account": "계좌 삭제 시 함께 삭제",
    "aggregate": "삭제 금지 (과거 기록)",
    "config": "화면에서만 편집",
    "reference": "배치가 다시 채움",
    "runtime": "TTL·배치가 정리",
    "personal": "화면에서만 편집",
}


@dataclass(frozen=True)
class TableSpec:
    """컬렉션 하나의 소유 관계와 설명.

    Attributes:
        name: 컬렉션 이름. `name_pattern` 이 있으면 표시용 이름이다.
        category: 분류(위 `CATEGORY_LABELS` 키).
        purpose: 이 테이블이 담는 것 — 화면 설명 칸.
        owner_field: 소유자 id 가 들어 있는 필드. 비면 소유자별로 나뉘지 않는다.
        owner_is_id: 소유자 id 가 `_id` 인 경우(문서 하나 = 소유자 하나).
        name_pattern: 컬렉션 **이름 자체**가 소유자에서 파생될 때의 형식(`cache_{owner}_stocks`).
            이 경우 풀·계좌가 늘어도 카탈로그를 고칠 필요가 없다.
        owner_note: 소유자 필드 이름이 실제 내용과 다를 때의 설명(옛 명명 잔재 등).
    """

    name: str
    category: str
    purpose: str
    owner_field: str = ""
    owner_is_id: bool = False
    name_pattern: str = ""
    owner_note: str = ""


# ── 종목풀이 소유하는 것 ────────────────────────────────────────────────────
_POOL_TABLES: tuple[TableSpec, ...] = (
    TableSpec("stock_meta", "pool", "종목풀에 등록된 종목(이름·상장일·버킷·메모)", owner_field="ticker_type"),
    TableSpec("stock_cache_meta", "pool", "종목별 계산 메타(배당률·보수·시총순위 등)", owner_field="ticker_type"),
    TableSpec(
        "previous_stock_cache_meta",
        "pool",
        "전일 기준 종목 메타 스냅샷 — 변화량 표시에 쓴다",
        owner_field="ticker_type",
    ),
    TableSpec("pool_rank_summary", "pool", "종목풀 순위 요약(화면 진입 시 즉시 표시용)", owner_is_id=True),
    TableSpec(
        "price_anomaly_alerts",
        "pool",
        "가격 캐시 갱신 중 발견한 이상 변동 기록",
        owner_field="account_id",
        owner_note="필드명은 account_id 지만 값은 종목풀(ticker_type) 이다 — 계좌와 풀이 같은 개념이던 시절의 잔재.",
    ),
    TableSpec(
        "cache_<종목풀>_stocks",
        "pool",
        "종목풀별 일봉 가격 캐시(백테스트·순위의 가격 소스)",
        name_pattern="cache_{owner}_stocks",
    ),
)

# ── 계좌가 소유하는 것 ──────────────────────────────────────────────────────
_ACCOUNT_TABLES: tuple[TableSpec, ...] = (
    TableSpec("account_notes", "account", "계좌별 메모", owner_field="account_id"),
    TableSpec(
        "cache_<계좌>_stocks",
        "account",
        "계좌 보유 종목의 일봉 가격 캐시",
        name_pattern="cache_{owner}_stocks",
    ),
)

# ── 집계·이력 — 소유자가 사라져도 지우지 않는다 ────────────────────────────
# 계좌를 지웠다고 과거 총자산 기록을 고쳐 쓰면 지난 수익률 그래프가 바뀐다.
# 이 분류에 있는 것은 삭제 로직이 **절대 건드리지 않는다**.
_AGGREGATE_TABLES: tuple[TableSpec, ...] = (
    TableSpec("daily_snapshots", "aggregate", "일자별 계좌 자산 스냅샷 — 사라진 계좌 기록도 그대로 둔다"),
    TableSpec("daily_fund_data", "aggregate", "일별 자금 집계(입출금·환율·비중)"),
    TableSpec("weekly_fund_data", "aggregate", "주별 자금 집계"),
    TableSpec("monthly_fund_data", "aggregate", "월별 자금 집계"),
    TableSpec("yearly_fund_data", "aggregate", "연별 자금 집계"),
    TableSpec("market_breadth_daily", "aggregate", "시장 폭(ADR) 일별 적립"),
)

# ── 설정 — 시스템의 단일 소스 ──────────────────────────────────────────────
_CONFIG_TABLES: tuple[TableSpec, ...] = (
    TableSpec("pool_settings", "config", "종목풀 정의와 설정(이평선·슬리피지·손절)", owner_is_id=True),
    TableSpec("account_settings", "config", "계좌 정의와 설정(합성 풀·알람·증권사 API)", owner_is_id=True),
    TableSpec("portfolio_master", "config", "계좌 원장 — 보유 수량·평단·현금"),
    TableSpec("system_config", "config", "전략 설정·알림 상태·토큰 등 시스템 키/값"),
    TableSpec("leverage_config", "config", "레버리지 추천 설정"),
)

# ── 참조 — 배치가 외부에서 받아 채운다. 지워도 다음 배치가 복구한다 ────────
_REFERENCE_TABLES: tuple[TableSpec, ...] = (
    TableSpec("index_constituents", "reference", "지수 구성종목(KOSPI200·KOSDAQ150·SP500·NDX100·ASX200)"),
    TableSpec("etf_market_master", "reference", "KIS 국내 ETF 마스터 캐시"),
    TableSpec("kor_dividend_stocks", "reference", "한국 배당주 화면의 종목별 재무·배당 지표"),
    TableSpec("yahoo_baseline_prices", "reference", "yfinance 기준가 캐시(전일 종가 대조용)"),
    TableSpec("cache_fx_stocks", "reference", "환율 일봉 캐시"),
    TableSpec("cache_etf_stocks", "reference", "레버리지 추천이 쓰는 지수·ETF 일봉 캐시"),
)

# ── 실행 상태 — 큐·락·진행 기록 ────────────────────────────────────────────
_RUNTIME_TABLES: tuple[TableSpec, ...] = (
    TableSpec("batch_queue", "runtime", "배치 실행 큐(24시간 TTL)"),
    TableSpec("batch_locks", "runtime", "배치 중복 실행 방지 락 · 배포 플래그"),
    TableSpec("cache_refresh_status", "runtime", "가격 캐시 갱신 완료 시각"),
    TableSpec("leverage_state", "runtime", "레버리지 알림 발송 상태"),
)

_PERSONAL_TABLES: tuple[TableSpec, ...] = (TableSpec("memos", "personal", "개인 메모장(할 일·목록)"),)


@dataclass(frozen=True)
class OwnedKeySpec:
    """설정 문서 **안쪽**에 소유자 id 가 키로 들어가는 자리.

    컬렉션 단위가 아니라 문서 하나의 하위 키라 컬렉션 목록만 봐서는 보이지 않는다.
    실제로 `new_high_notify_state.pools` 에 삭제된 풀이 남아 있었다.
    """

    doc_id: str
    path: str
    owner: str  # "pool" | "account"
    purpose: str


# 설정 문서 안에 소유자 id 가 **키**로 들어가는 자리 (전부 system_config 컬렉션).
OWNED_KEY_SPECS: tuple[OwnedKeySpec, ...] = (
    OwnedKeySpec("momentum_settings", "settings.settings_by_pool", "pool", "모멘텀 전략의 풀별 저장 설정"),
    OwnedKeySpec("new_high_settings", "settings_by_pool", "pool", "신고가 전략의 풀별 저장 설정"),
    OwnedKeySpec("new_high_notify_state", "pools", "pool", "신고가 알림 발송 상태(풀별)"),
    OwnedKeySpec("strategy_mix_notify_state", "accounts", "account", "합성 액션 알림 발송 상태(계좌별)"),
    OwnedKeySpec("broker_balance_sync_state", "accounts", "account", "증권사 잔고 동기화 실패/복구 상태(계좌별)"),
)


@dataclass(frozen=True)
class OwnerRefSpec:
    """다른 문서가 소유자를 **값으로 가리키는** 자리. 소유자가 사라지면 참조를 비워야 한다."""

    collection: str
    field: str
    owner: str  # "pool" | "account"
    purpose: str


OWNER_REF_SPECS: tuple[OwnerRefSpec, ...] = (
    OwnerRefSpec("account_settings", "mix_sm_pool", "pool", "합성 모멘텀 슬리브가 쓰는 종목풀"),
    OwnerRefSpec("account_settings", "mix_nh_pool", "pool", "합성 신고가 슬리브가 쓰는 종목풀"),
)

# 종목풀·계좌가 아니면서 소유자 자리에 들어가는 **예약 이름**.
# 환율(fx)·지수/레버리지 ETF(etf) 캐시가 같은 저장 경로를 쓰기 때문에 생긴다.
# 주인이 없다고 지우면 환율·레버리지 데이터가 통째로 사라지므로 고아로 보지 않는다.
RESERVED_OWNERS: frozenset[str] = frozenset({"fx", "etf"})

TABLE_SPECS: tuple[TableSpec, ...] = (
    *_POOL_TABLES,
    *_ACCOUNT_TABLES,
    *_AGGREGATE_TABLES,
    *_CONFIG_TABLES,
    *_REFERENCE_TABLES,
    *_RUNTIME_TABLES,
    *_PERSONAL_TABLES,
)

# 화면 그룹 순서 — 소유자별 → 보존 → 나머지.
CATEGORY_ORDER: tuple[str, ...] = ("pool", "account", "aggregate", "config", "reference", "runtime", "personal")


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패해 테이블 목록을 읽을 수 없습니다.")
    return db


def _collection_stats(db, name: str) -> dict[str, int]:
    """문서 수·데이터 크기·인덱스 크기. 실패하면 문서 수만이라도 센다."""
    try:
        stats = list(db[name].aggregate([{"$collStats": {"storageStats": {}}}]))[0]["storageStats"]
        return {
            "count": int(stats.get("count") or 0),
            "size": int(stats.get("size") or 0),
            "index_size": int(stats.get("totalIndexSize") or 0),
        }
    except Exception as exc:
        logger.warning("컬렉션 통계 조회 실패 (%s): %s", name, exc)
        try:
            return {"count": int(db[name].count_documents({})), "size": 0, "index_size": 0}
        except Exception:
            return {"count": 0, "size": 0, "index_size": 0}


@dataclass
class _PatternMatch:
    """이름이 소유자에서 파생되는 컬렉션 하나의 실제 인스턴스."""

    name: str
    owner: str
    stats: dict[str, int] = field(default_factory=dict)


def _pattern_instances(existing: set[str], spec: TableSpec) -> list[_PatternMatch]:
    """`cache_{owner}_stocks` 형식에 실제로 존재하는 컬렉션을 소유자별로 뽑는다."""
    prefix, suffix = spec.name_pattern.split("{owner}")
    matches: list[_PatternMatch] = []
    for name in existing:
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        owner = name[len(prefix) : len(name) - len(suffix)]
        if not owner or "_tmp_" in owner:
            continue
        matches.append(_PatternMatch(name=name, owner=owner))
    return sorted(matches, key=lambda m: m.owner)


def _owner_values(db, spec: TableSpec) -> list[str]:
    """그 컬렉션에 실제로 들어 있는 소유자 id 목록."""
    field_name = "_id" if spec.owner_is_id else spec.owner_field
    if not field_name:
        return []
    try:
        return [str(v).strip() for v in db[spec.name].distinct(field_name) if v is not None and str(v).strip()]
    except Exception as exc:
        logger.warning("소유자 값 조회 실패 (%s.%s): %s", spec.name, field_name, exc)
        return []


def _dig(doc: dict[str, Any], path: str) -> dict[str, Any]:
    """`a.b.c` 경로를 따라 내려간 dict. 중간이 없으면 빈 dict."""
    node: Any = doc
    for part in path.split("."):
        node = (node or {}).get(part)
        if not isinstance(node, dict):
            return {}
    return node


def scan_orphans() -> dict[str, Any]:
    """살아있는 종목풀·계좌와 대조해 **주인 없는 데이터**를 찾는다.

    삭제와 같은 카탈로그를 쓰므로, 카탈로그에 한 줄을 더하면 점검도 함께 커진다.
    반환은 자리별 목록 — 화면이 그대로 보여주고, 정리도 이 목록을 기준으로 한다.
    """
    from utils.settings_loader import list_available_accounts, list_available_ticker_types

    db = _db()
    live = {"pool": set(list_available_ticker_types()), "account": set(list_available_accounts())}
    existing = {n for n in db.list_collection_names() if "_tmp_" not in n}

    docs: list[dict[str, Any]] = []
    for spec in TABLE_SPECS:
        if spec.category not in ("pool", "account") or spec.name_pattern:
            continue
        dead = sorted(v for v in _owner_values(db, spec) if v not in live[spec.category] and v not in RESERVED_OWNERS)
        if not dead:
            continue
        field_name = "_id" if spec.owner_is_id else spec.owner_field
        docs.append(
            {
                "location": spec.name,
                "owner_kind": spec.category,
                "detail": f"{field_name} ∈ {', '.join(dead)}",
                "owners": dead,
                "count": db[spec.name].count_documents({field_name: {"$in": dead}}),
            }
        )

    # 이름이 소유자에서 파생되는 컬렉션 — 소유자가 풀·계좌 어디에도 없으면 통째로 고아다.
    # 카탈로그에 이름으로 직접 등록된 컬렉션(cache_fx_stocks 등)은 형식이 같아도 제외한다.
    named = {spec.name for spec in TABLE_SPECS if not spec.name_pattern}
    collections: list[dict[str, Any]] = []
    for spec in TABLE_SPECS:
        if not spec.name_pattern or spec.category != "pool":
            continue  # 같은 형식을 풀·계좌가 공유하므로 한 번만 훑는다
        for match in _pattern_instances(existing, spec):
            if match.name in named or match.owner in RESERVED_OWNERS:
                continue
            if match.owner in live["pool"] or match.owner in live["account"]:
                continue
            stats = _collection_stats(db, match.name)
            collections.append(
                {
                    "location": match.name,
                    "owner_kind": "unknown",
                    "detail": f"소유자 '{match.owner}' 가 종목풀·계좌 어디에도 없음",
                    "owners": [match.owner],
                    "count": stats["count"],
                    "size": stats["size"],
                }
            )

    keys: list[dict[str, Any]] = []
    for key_spec in OWNED_KEY_SPECS:
        doc = db["system_config"].find_one({"_id": key_spec.doc_id}) or {}
        dead = sorted(k for k in _dig(doc, key_spec.path) if k not in live[key_spec.owner])
        if dead:
            keys.append(
                {
                    "location": f"system_config › {key_spec.doc_id}.{key_spec.path}",
                    "owner_kind": key_spec.owner,
                    "detail": f"키 {', '.join(dead)}",
                    "owners": dead,
                    "count": len(dead),
                }
            )

    refs: list[dict[str, Any]] = []
    for ref in OWNER_REF_SPECS:
        dead_docs = [
            str(d["_id"])
            for d in db[ref.collection].find({ref.field: {"$nin": [None, ""]}}, {ref.field: 1})
            if str(d.get(ref.field) or "").strip() not in live[ref.owner]
        ]
        if dead_docs:
            refs.append(
                {
                    "location": f"{ref.collection}.{ref.field}",
                    "owner_kind": ref.owner,
                    "detail": f"사라진 소유자를 가리키는 문서: {', '.join(dead_docs)}",
                    "owners": dead_docs,
                    "count": len(dead_docs),
                }
            )

    items = docs + collections + keys + refs
    return {"items": items, "total": sum(int(i["count"]) for i in items)}


def build_data_table_payload() -> dict[str, Any]:
    """화면용 페이로드 — 분류별 테이블 목록과 실측 통계.

    카탈로그에 없는 컬렉션은 `unclassified` 로 따로 담는다(등록 누락을 드러내기 위해).
    """
    from utils.settings_loader import list_available_accounts, list_available_ticker_types

    db = _db()
    existing = {n for n in db.list_collection_names() if "_tmp_" not in n}
    pools = set(list_available_ticker_types())
    accounts = set(list_available_accounts())

    known: set[str] = set()
    rows: list[dict[str, Any]] = []

    for spec in TABLE_SPECS:
        base = {
            "name": spec.name,
            "category": spec.category,
            "category_label": CATEGORY_LABELS.get(spec.category, spec.category),
            "policy": CATEGORY_POLICIES.get(spec.category, ""),
            "purpose": spec.purpose,
            "owner_field": "_id" if spec.owner_is_id else spec.owner_field,
            "owner_note": spec.owner_note,
        }
        if spec.name_pattern:
            # 소유자별 컬렉션 — 실제로 있는 것만 펼쳐서 각각 한 행으로 보여준다.
            owners = pools if spec.category == "pool" else accounts
            for match in _pattern_instances(existing, spec):
                # 같은 형식을 풀·계좌가 나눠 쓰므로, 이 분류의 소유자 목록에 있는 것만 가져간다.
                if match.owner not in owners and match.name in known:
                    continue
                if match.owner not in owners:
                    continue
                known.add(match.name)
                rows.append({**base, "name": match.name, "owner": match.owner, **_collection_stats(db, match.name)})
            continue

        known.add(spec.name)
        if spec.name not in existing:
            # 카탈로그에는 있는데 DB 에 없다 — 아직 안 만들어졌거나 이름이 바뀐 것이다.
            rows.append({**base, "missing": True, "count": 0, "size": 0, "index_size": 0})
            continue
        rows.append({**base, **_collection_stats(db, spec.name)})

    unclassified = [
        {
            "name": name,
            "category": "unclassified",
            "category_label": "미분류",
            "policy": "카탈로그에 등록되지 않음",
            "purpose": "",
            "owner_field": "",
            "owner_note": "",
            **_collection_stats(db, name),
        }
        for name in sorted(existing - known)
    ]

    return {
        "rows": rows,
        "unclassified": unclassified,
        # 주인 없는 데이터 — 같은 카탈로그로 찾는다(삭제와 점검이 갈리지 않게).
        "orphans": scan_orphans(),
        "category_order": list(CATEGORY_ORDER),
        "category_labels": dict(CATEGORY_LABELS),
        "totals": {
            "collections": len(rows) + len(unclassified),
            "count": sum(int(r.get("count") or 0) for r in rows + unclassified),
            "size": sum(int(r.get("size") or 0) for r in rows + unclassified),
            "index_size": sum(int(r.get("index_size") or 0) for r in rows + unclassified),
        },
        "pools": sorted(pools),
        "accounts": sorted(accounts),
    }
