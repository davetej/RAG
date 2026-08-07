import logging
from typing import Optional


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the application process."""
    root = logging.getLogger()
    if root.handlers:
        return

    numeric_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else __name__)
