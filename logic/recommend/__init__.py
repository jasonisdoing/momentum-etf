"""Shallow exports for signals logic package with lazy imports.

Avoid importing heavy submodules at package import time to prevent circular imports.
"""

from __future__ import annotations

from typing import Any, Dict


def main() -> None:
    from .pipeline import main as _main

    return _main()


def generate_signal_report(*args: Any, **kwargs: Dict[str, Any]):
    from .pipeline import generate_country_signal_report as _gen

    return _gen(*args, **kwargs)


def calculate_benchmark_comparison(*args: Any, **kwargs: Dict[str, Any]):
    from .benchmarks import calculate_benchmark_comparison as _calc

    return _calc(*args, **kwargs)


__all__ = ["main", "generate_signal_report", "calculate_benchmark_comparison"]
