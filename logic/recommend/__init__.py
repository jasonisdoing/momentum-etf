"""Shallow exports for signals logic package with lazy imports.

Avoid importing heavy submodules at package import time to prevent circular imports.
"""

from __future__ import annotations

from typing import Any, Dict


def main() -> None:
    from .pipeline import main as _main

    return _main()


def generate_recommendation_report(*args: Any, **kwargs: Dict[str, Any]):
    from .pipeline import generate_country_recommendation_report as _gen

    return _gen(*args, **kwargs)


__all__ = ["main", "generate_recommendation_report"]
