import logging

logging.getLogger("tensorflow").setLevel(logging.FATAL)

import absl.logging

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False


import sys
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import cast

import rich
from rich.logging import RichHandler
from rich.text import Text


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    # redundant, but useful for type hinting
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"

    def to_logging_level(self):
        return {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
            LogLevel.debug: logging.DEBUG,
            LogLevel.info: logging.INFO,
            LogLevel.warning: logging.WARNING,
            LogLevel.error: logging.ERROR,
            LogLevel.critical: logging.CRITICAL,
        }[self]


def remove_color_tags(log_message):
    text = Text.from_markup(log_message)
    return text.plain


class CustomLogger(logging.Logger):
    def normalize_msg(self, msg, tab=0, tab_char="==", **kwargs):
        msg = str(msg)
        msg = dedent(msg)
        if tab > 0:
            msg = f"{tab_char * tab} {msg}"
        return msg

    def info(self, msg, *args, tab=0, tab_char="==", raw_print: bool = False, **kwargs):
        msg = self.normalize_msg(msg, tab, tab_char, **kwargs)
        if raw_print:
            msg = msg % args
            rich.print(msg)
            return

        super().info(msg, *args, **kwargs)

    def debug(self, msg, *args, tab=0, tab_char="==", raw_print: bool = False, **kwargs):
        msg = self.normalize_msg(msg, tab, tab_char, **kwargs)
        if raw_print:
            msg = msg % args
            rich.print(msg)
            return

        super().debug(msg, *args, **kwargs)

    def warning(self, msg, *args, tab=0, tab_char="==", raw_print: bool = False, **kwargs):
        msg = self.normalize_msg(msg, tab, tab_char, **kwargs)
        if raw_print:
            msg = msg % args
            rich.print(msg)
            return

        super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, tab=0, tab_char="==", raw_print: bool = False, **kwargs):
        msg = self.normalize_msg(msg, tab, tab_char, **kwargs)
        if raw_print:
            msg = msg % args
            rich.print(msg)
            return

        super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, tab=0, tab_char="==", raw_print: bool = False, **kwargs):
        msg = self.normalize_msg(msg, tab, tab_char, **kwargs)
        if raw_print:
            msg = msg % args
            rich.print(msg)
            return

        super().critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, tab=0, tab_char="==", raw_print: bool = False, **kwargs):
        msg = self.normalize_msg(msg, tab, tab_char, **kwargs)
        if raw_print:
            msg = msg % args
            rich.print(msg)
            return

        super().log(level, msg, *args, **kwargs)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=3):
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)


# https://stackoverflow.com/a/56944256
class ColoredFormatter(logging.Formatter):
    # _grey = "\x1b[38;20m"
    _green = "\x1b[32;20m"
    _purple = "\x1b[35;20m"
    _yellow = "\x1b[33;20m"
    _red = "\x1b[31;20m"
    _bold_red = "\x1b[31;1m"
    _reset = "\x1b[0m"
    _blue = "\x1b[34;20m"
    # _format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    # _lvl_format = "%(levelname)s\t"
    # get first letter of levelname
    # _prefix_format = "[%(levelname).1s,%(filename)s:%(lineno)d] "
    _prefix_format = "[%(levelname).1s] "
    _format = "%(message)s"

    FORMATS = {
        logging.DEBUG: _purple + _prefix_format + _reset + _format,
        logging.INFO: _blue + _prefix_format + _reset + _format,
        logging.WARNING: _yellow + _prefix_format + _reset + _format,
        logging.ERROR: _red + _prefix_format + _reset + _format,
        logging.CRITICAL: _bold_red + _prefix_format + _reset + _format,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class RichMarkupFilter(logging.Filter):
    def filter(self, record):
        record.msg = remove_color_tags(record.msg)
        return True


def _create_rich_handler(width: int | None = None) -> RichHandler:
    import click
    import typer
    from rich.console import Console

    return RichHandler(markup=True, rich_tracebacks=True, tracebacks_suppress=[click, typer], console=Console(width=width))


def log_global_stdout_setup(console_width: int | None = None, level: LogLevel = LogLevel.INFO):
    logging.setLoggerClass(CustomLogger)
    logging.root.handlers = []
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(ColoredFormatter())
    # stdout_handler = _create_rich_handler(width=console_width)

    logging.basicConfig(
        format="%(message)s",
        # datefmt="%H:%M:%S",
        datefmt="[%X]",
        level=level.to_logging_level(),
        handlers=[
            stdout_handler,
        ],
    )


def log_global_setup(output_path: Path | str | None = None, level: LogLevel = LogLevel.INFO, console_width: int | None = None) -> CustomLogger:
    logging.setLoggerClass(CustomLogger)
    logging.root.handlers = []
    handlers = []
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(ColoredFormatter())
    # stdout_handler = _create_rich_handler(width=console_width)
    handlers.append(stdout_handler)

    if output_path is not None:
        output_path = Path(output_path)
        file_handler = logging.FileHandler(output_path, mode="w")
        file_handler.addFilter(RichMarkupFilter())
        handlers.append(file_handler)

    logging.basicConfig(
        # format="%(message)s (%(filename)s:%(lineno)d)",
        format="%(message)s",
        # datefmt="%H:%M:%S",
        datefmt="[%X]",
        level=level.to_logging_level(),
        handlers=handlers,
    )
    log = logging.getLogger("root")
    log.__class__ = CustomLogger
    return cast(CustomLogger, log)


def log_create(
    name: str | None = None, output_path: Path | str | None = None, level: LogLevel = LogLevel.INFO, console_width: int | None = None
) -> CustomLogger:
    logging.setLoggerClass(CustomLogger)
    # create logger
    logging.basicConfig(
        format="%(message)s",
        # datefmt="%H:%M:%S",
        datefmt="[%X]",
        level=level.to_logging_level(),
    )
    logger = logging.getLogger(name)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(ColoredFormatter())
    # stdout_handler = _create_rich_handler(width=console_width)

    logger.setLevel(level.to_logging_level())
    if output_path:
        file_handler = logging.FileHandler(output_path, mode="w")
        logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return cast(CustomLogger, logger)


def log_get(name: str | None = None) -> CustomLogger:
    return cast(CustomLogger, logging.getLogger(name))


log_global_stdout_setup()
logging.setLoggerClass(CustomLogger)

__all__ = [
    "LogLevel",
    "log_global_setup",
    "log_global_stdout_setup",
    "log_create",
    "log_get",
]
