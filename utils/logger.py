"""Logging setup shared by the magma ocean model, its drivers, and its tests."""
import logging
import sys

LOGGER_NAME = 'magma_ocean'
_FORMAT = '%(asctime)s - %(levelname)-8s - %(name)s: %(message)s'


def get_logger(name=None):
    """
    Return the package logger or one of its children.

    Parameters
    ----------
    name : str, optional
        Child name (e.g. a module's ``__name__``). ``None`` returns the package logger.

    Returns
    -------
    logging.Logger
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f'{LOGGER_NAME}.{name}')


def setup_logging(level='INFO', log_file=None, console=True):
    """
    Configure the package logger to print to the console and, optionally, to a file.

    Calling this again replaces the handlers installed by a previous call, so drivers that run many models in one
    process can redirect each run to its own log file.

    Parameters
    ----------
    level : str or int, optional
        Logging level for all handlers.
    log_file : str or path-like, optional
        If given, log messages are also written (UTF-8, append mode) to this file.
    console : bool, optional
        If True, log messages are printed to stderr.

    Returns
    -------
    logging.Logger
        The configured package logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    # Do not pass records to the root logger; TidalPy installs its own root-level handlers.
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT)
    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
