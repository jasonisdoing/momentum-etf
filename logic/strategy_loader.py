"""동적으로 전략 모듈을 로드하는 헬퍼."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

from config import SELECTED_STRATEGY


def _strategy_package_name() -> str:
    strategy = (SELECTED_STRATEGY or "").strip()
    if not strategy:
        raise ValueError("data.settings.common.SELECTED_STRATEGY 값이 비어 있습니다.")
    return f"strategies.{strategy}"


@lru_cache(maxsize=1)
def get_strategy_package():
    """선택된 전략 패키지를 반환합니다."""

    return import_module(_strategy_package_name())


@lru_cache(maxsize=1)
def get_backtest_module():
    return import_module(f"{_strategy_package_name()}.backtest")


@lru_cache(maxsize=1)
def get_recommend_module():
    return import_module(f"{_strategy_package_name()}.recommend")


@lru_cache(maxsize=1)
def get_constants_module():
    return import_module(f"{_strategy_package_name()}.constants")


@lru_cache(maxsize=1)
def get_rules_module():
    return import_module(f"{_strategy_package_name()}.rules")


@lru_cache(maxsize=1)
def get_rules_class():
    module = get_rules_module()
    if not hasattr(module, "StrategyRules"):
        raise AttributeError(f"{module.__name__} 모듈에 StrategyRules 클래스가 없습니다.")
    return getattr(module, "StrategyRules")


__all__ = [
    "get_strategy_package",
    "get_backtest_module",
    "get_recommend_module",
    "get_constants_module",
    "get_rules_module",
    "get_rules_class",
]
