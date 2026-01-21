"""Pipelines = end-to-end workflows.

Each pipeline exposes a `run(ctx, ...)` function.
"""

from pipelines.forecast_channel import forecast_channel

__all__ = ["forecast_channel"]
