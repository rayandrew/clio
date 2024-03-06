from contextlib import contextmanager
from timeit import default_timer as base_timer


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

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed():.4f} sec"


@contextmanager
def timer(name: str):
    start = base_timer()
    yield
    end = base_timer()
    print(f"{name}: {end - start:.4f} sec")


__all__ = ["Timer", "timer"]
