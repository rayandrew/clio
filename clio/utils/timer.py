from contextlib import contextmanager
from functools import wraps
from timeit import default_timer as base_timer
from typing import Any, Callable

from clio.utils.logging import log_get


class Timer:
    """
    A simple timer class.
    """

    def __init__(self, name: str = "Timer", start: float = 0.0, end: float = 0.0):
        self.start = 0.0
        self.end = 0.0
        self.name = name

    def __enter__(self):
        self.start = base_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = base_timer()

    @property
    def elapsed(self) -> float:
        """
        Retrieves the elapsed time.

        :return: The elapsed time, as a float.
        """
        return self.end - self.start

    def __repr__(self) -> str:
        return f"{self.name}: {self.elapsed:.4f} sec"

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed:.4f} sec"


default_timer = base_timer


# create decorator to time functions
def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def timed(*args: tuple, **kwargs: dict) -> Any:
        log = log_get("timeit_" + func.__name__)
        res = None
        with Timer(func.__name__) as t:
            res = func(*args, **kwargs)
        log.info("Elapsed time of %s: %.4f ms", func.__name__, t.elapsed * 1000)
        return res

    return timed


__all__ = ["Timer", "default_timer"]
