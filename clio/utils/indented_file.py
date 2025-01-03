import io
from contextlib import contextmanager
from pathlib import Path


class IndentedFile:
    def __init__(self, file_path: str | Path | io.TextIOWrapper, indent: int = 0, indent_str: str = " " * 4, buffer_size: int = -1):
        self.buffer_size = buffer_size
        if isinstance(file_path, io.IOBase):
            self.file = file_path
            self.file_path = None
            self.opened = True
        else:
            self.file_path = file_path
            self.file = None
            self.opened = False
            self.open()
        self.prev_indent = indent
        self.indent = indent
        self.indent_str = indent_str

    def open(self):
        if self.opened:
            return self

        if self.file_path is None:
            raise ValueError("file_path is None")

        self.file = open(self.file_path, "w", buffering=self.buffer_size)
        self.opened = True

        return self

    def close(self):
        if not self.opened and self.file is None:
            return self
        if self.file is not None:
            self.file.close()
        return self

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def inc_indent(self):
        self.prev_indent = self.indent
        self.indent += 1

    def dec_indent(self):
        self.prev_indent = self.indent
        self.indent -= 1

    def add_indent(self, indent: int):
        self.prev_indent = self.indent
        self.indent += indent

    def sub_indent(self, indent: int):
        self.prev_indent = self.indent
        self.indent -= indent

    def set_indent(self, indent: int):
        self.prev_indent = self.indent
        self.indent = indent

    def revert_indent(self):
        self.indent = self.prev_indent

    @contextmanager
    def section(self, section: str, *args):
        self.writeln(section, *args)
        self.inc_indent()
        try:
            yield
        finally:
            self.dec_indent()

    @contextmanager
    def block(self, indent: int = 1):
        if indent <= 0:
            raise ValueError("Indent must be greater than 0")
        self.add_indent(indent)

        try:
            yield
        finally:
            self.sub_indent(indent)

    def write(self, msg: str, *args):
        if not self.opened or self.file is None:
            raise ValueError("File is not opened")
        if args:
            msg = msg % args
        self.file.write(self.indent * self.indent_str + msg)

    def writeln(self, msg: str, *args):
        self.write(msg, *args)
        if self.file is not None:
            self.file.write("\n")

    def writelines(self, lines: list[str]):
        for line in lines:
            self.writeln(line)

    def flush(self):
        assert self.file is not None, "File is not opened"
        self.file.flush()

    def __del__(self):
        self.close()
        self.file = None
        self.file_path = None
        self.opened = False
        self.indent = 0
        self.indent_str = " "

    def __str__(self):
        return f"IndentedFile(file_path={self.file_path}, indent={self.indent}, indent_str={self.indent_str})"


__all__ = ["IndentedFile"]
