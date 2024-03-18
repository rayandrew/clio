from pathlib import Path


# https://stackoverflow.com/a/49782093/2418586
def rmdir(directory: Path) -> None:
    if not directory.exists():
        return
    if directory.is_file():
        directory.unlink()
        return
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


__all__ = ["rmdir"]
