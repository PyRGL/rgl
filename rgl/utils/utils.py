import logging


def get_logger(format="s"):  # s|ms|path
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    if format == "s":
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    elif format == "ms":
        formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    elif format == "path":
        formatter = logging.Formatter("[%(asctime)s] [%(pathname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent messages from being passed to ancestor loggers
    return logger
