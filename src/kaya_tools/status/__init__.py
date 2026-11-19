"""The `kaya status` report: GPU nodes per partition, the queue, and your recent jobs."""

from .report import handle_status

__all__ = ["handle_status"]
