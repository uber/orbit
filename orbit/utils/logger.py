import logging
from typing import Optional


def make_info_logger(name: str, path: Optional[str] = None) -> logging.Logger:
    """generate new logger in a standardized way
    Returns:
        logging.Logger: logger created in INFO level and time-stamp
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if path is None:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        fh = logging.FileHandler(path, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name=name)
    if len(logger.handlers) == 0:
        logger = make_info_logger(name)
    return logger
