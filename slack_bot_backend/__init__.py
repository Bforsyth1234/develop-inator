"""Slack bot backend package."""

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    if name == "app":
        from .main import app

        return app
    if name == "create_app":
        from .main import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
