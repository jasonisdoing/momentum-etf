# Aggregated exports for signals logic package
from .pipeline import main, generate_signal_report
from .benchmarks import calculate_benchmark_comparison

__all__ = [
    "main",
    "generate_signal_report",
    "calculate_benchmark_comparison",
]
