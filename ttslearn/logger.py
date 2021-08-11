import logging
from logging import Logger
from pathlib import Path
from typing import Dict, Optional

format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"

_initialized: Dict[str, bool] = {}


def getLogger(
    verbose: int = 0,
    filename: Optional[str] = None,
    name: str = "ttslearn",
    add_stream_handler: bool = True,
) -> Logger:
    """Get a logger instance.

    Args:
        verbose (int): Verbosity level. Can be a number from -1 (low) to 10 (high).
        filename (str, optional): Name of the file to log to.
        name (str, optional): Name of the logger.
        add_stream_handler (bool, optional): Add a stream handler to the logger.

    Returns:
        logging.Logger: Logger instance.

    Examples:

        >>> from ttslearn.logger import getLogger
        >>> logger = getLogger(verbose=10)
        >>> logger.info("This is a test")
        2021-07-24 10:20:52,924 (ttslearn:23) INFO: This is a test
        >>> logger.debug("This is a debug message")
        2021-07-24 10:20:52,924 (ttslearn:23) DEBUG: This is a debug message
        >>> logger.warning("This is a warning message")
        2021-07-24 10:20:52,924 (ttslearn:23) WARNING: This is a warning message
    """
    global _initialized
    logger = logging.getLogger(name)
    if verbose >= 10:
        logger.setLevel(logging.DEBUG)
    elif verbose > 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARN)

    if _initialized.get(name, False):
        return logger
    else:
        _initialized[name] = True

    if add_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(format))
        logger.addHandler(stream_handler)

    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(format))
        logger.addHandler(file_handler)

    return logger
