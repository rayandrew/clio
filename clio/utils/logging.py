import logging
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import cast

import rich
from rich.logging import RichHandler
from rich.text import Text


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


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


class RichMarkupFilter(logging.Filter):
    def filter(self, record):
        record.msg = remove_color_tags(record.msg)
        return True


def _create_rich_handler(width: int | None = None) -> RichHandler:
    import click
    import typer
    from rich.console import Console

    return RichHandler(markup=True, rich_tracebacks=True, tracebacks_suppress=[click, typer], console=Console(width=width))


def log_global_stdout_setup(console_width: int | None = None):
    logging.setLoggerClass(CustomLogger)
    logging.root.handlers = []
    stdout_handler = _create_rich_handler(width=console_width)

    logging.basicConfig(
        format="%(message)s",
        # datefmt="%H:%M:%S",
        datefmt="[%X]",
        level=logging.INFO,
        handlers=[
            stdout_handler,
        ],
    )


def log_global_setup(output_path: Path | str | None = None, level: LogLevel = LogLevel.INFO, console_width: int | None = None) -> CustomLogger:
    logging.setLoggerClass(CustomLogger)
    logging.root.handlers = []
    handlers = []
    stdout_handler = _create_rich_handler(width=console_width)
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
        level=level.value,
        handlers=handlers,
    )
    log = logging.getLogger("root")
    log.__class__ = CustomLogger
    return cast(CustomLogger, log)


def log_create(output_path: Path, level: LogLevel = LogLevel.INFO, name: str | None = None, console_width: int | None = None) -> CustomLogger:
    logging.setLoggerClass(CustomLogger)
    # create logger
    logging.basicConfig(
        format="%(message)s",
        # datefmt="%H:%M:%S",
        datefmt="[%X]",
        level=level.value,
    )
    logger = logging.getLogger(name)
    stdout_handler = _create_rich_handler(width=console_width)

    file_handler = logging.FileHandler(output_path, mode="w")
    logger.setLevel(level.value)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return cast(CustomLogger, logger)


def log_get(name: str | None = None):
    return logging.getLogger(name)


logging.setLoggerClass(CustomLogger)
log_global_stdout_setup()

__all__ = [
    "LogLevel",
    "log_global_setup",
    "log_global_stdout_setup",
    "log_create",
    "log_get",
]
