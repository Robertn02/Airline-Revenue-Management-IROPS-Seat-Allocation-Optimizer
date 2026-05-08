"""Structured logging for Reroute.

Uses `rich` for human-readable terminal output. Library code uses module-level
loggers; only the CLI configures handlers.

Author: Phuc Nguyen
"""
from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


_CONFIGURED = False


def configure_logging(level: str = "INFO", quiet: bool = False) -> None:
    """Configure root logger with rich formatting. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    if quiet:
        level = "WARNING"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger."""
    return logging.getLogger(name)
