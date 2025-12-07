from typing import Any


def generate_recommendation_report(*args: Any, **kwargs: dict[str, Any]):
    from .pipeline import generate_account_recommendation_report as _gen

    return _gen(*args, **kwargs)


# Lazy import helper for RecommendationReport to avoid circular deps if any
def __getattr__(name: str):
    if name == "RecommendationReport":
        from .pipeline import RecommendationReport

        return RecommendationReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["generate_recommendation_report", "RecommendationReport"]
