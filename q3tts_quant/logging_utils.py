from __future__ import annotations

import json
import logging
from typing import Any


LOGGER = logging.getLogger("Q3TTS")


def configure_logging(level: int = logging.INFO) -> None:
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[Q3TTS] %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def log_event(event: str, **fields: Any) -> None:
    configure_logging()
    LOGGER.info("%s %s", event, json.dumps(fields, sort_keys=True, default=str))

