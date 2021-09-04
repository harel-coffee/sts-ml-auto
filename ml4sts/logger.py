# Imports: standard library
import os
import sys
import errno
import logging
from logging import config as logging_config


def load_config(log_level, log_dir, log_file_basename):

    try:
        os.makedirs(log_dir)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise err

    logger = logging.getLogger(__name__)

    log_file = f"{log_dir}/{log_file_basename}.txt"

    try:
        logging_config.dictConfig(_create_config(log_level, log_file))
        logging.info(f"Logging configuration loaded. Log will be saved to {log_file}")
    except Exception as err:
        logger.error("Failed to load logging config!")
        raise err


def _create_config(log_level, log_file):
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": (
                    "%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": sys.stdout,
            },
            "file": {
                "level": log_level,
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": log_file,
                "mode": "w",
            },
        },
        "loggers": {"": {"handlers": ["console", "file"], "level": log_level}},
    }
