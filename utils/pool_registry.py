"""종목풀(zpools) 설정/디렉토리 로더."""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

POOL_ROOT = Path(__file__).resolve().parents[1] / "zpools"
POOL_DIR_PATTERN = re.compile(r"^(?P<order>\d+)_(?P<pool>[a-z0-9_]+)$")


class PoolSettingsError(RuntimeError):
    """종목풀 설정 로딩 오류."""


def parse_pool_dir_name(dir_name: str) -> tuple[int, str]:
    normalized = (dir_name or "").strip().lower()
    match = POOL_DIR_PATTERN.fullmatch(normalized)
    if not match:
        raise PoolSettingsError(f"종목풀 디렉토리명은 '<order>_<pool>' 형식이어야 합니다: {dir_name}")
    return int(match.group("order")), match.group("pool")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        raise PoolSettingsError(f"설정 파일을 읽을 수 없습니다: {path}") from exc
    if not isinstance(data, dict):
        raise PoolSettingsError(f"설정 파일 루트는 객체(JSON object)여야 합니다: {path}")
    return data


def _iter_pool_dirs() -> list[tuple[str, Path]]:
    if not POOL_ROOT.exists():
        return []

    pool_dirs: dict[str, Path] = {}
    for item in POOL_ROOT.iterdir():
        if not item.is_dir() or item.name.startswith(".") or item.name.startswith("_"):
            continue
        _, pool_id = parse_pool_dir_name(item.name)
        pool_dirs[pool_id] = item

    return sorted(pool_dirs.items(), key=lambda pair: parse_pool_dir_name(pair[1].name)[0])


def list_available_pools() -> list[str]:
    return [pool_id for pool_id, _ in _iter_pool_dirs()]


@cache
def get_pool_dir(pool_id: str) -> Path:
    pool_norm = (pool_id or "").strip().lower()
    if not pool_norm:
        raise PoolSettingsError("종목풀 식별자를 지정해야 합니다.")

    pool_dirs = dict(_iter_pool_dirs())
    path = pool_dirs.get(pool_norm)
    if path is None:
        raise PoolSettingsError(f"종목풀 '{pool_norm}' 디렉토리를 찾을 수 없습니다.")
    return path


def load_pool_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    for pool_id, path in _iter_pool_dirs():
        order, _ = parse_pool_dir_name(path.name)
        config_path = path / "config.json"
        config_data: dict[str, Any] = {}
        if config_path.exists():
            try:
                config_data = _load_json(config_path)
            except PoolSettingsError:
                config_data = {}

        name = str(config_data.get("name") or pool_id)
        desc = str(config_data.get("desc") or "")

        configs.append(
            {
                "pool_id": pool_id,
                "order": order,
                "name": name,
                "desc": desc,
                "path": path,
                "config": config_data,
            }
        )

    return sorted(configs, key=lambda item: (item["order"], item["pool_id"]))


__all__ = [
    "PoolSettingsError",
    "POOL_ROOT",
    "parse_pool_dir_name",
    "list_available_pools",
    "get_pool_dir",
    "load_pool_configs",
    "get_pool_country_code",
]


def get_pool_country_code(pool_id: str, default: str = "kor") -> str:
    """종목풀 config.json에서 rank.country 값을 읽어 국가 코드를 반환한다."""
    pool_dir = get_pool_dir(pool_id)
    config_path = pool_dir / "config.json"
    if not config_path.exists():
        return default
    try:
        config_data = _load_json(config_path)
    except Exception:
        return default

    rank_cfg = config_data.get("rank")
    if isinstance(rank_cfg, dict):
        country = str(rank_cfg.get("country") or "").strip().lower()
        if country:
            return country
    return default
