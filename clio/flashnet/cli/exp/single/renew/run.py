import typer

from .each_window_use_recent import app as each_window_use_recent
from .threshold_use_recent import app as threshold_use_recent

app = typer.Typer()
app.add_typer(each_window_use_recent, name="each-window-use-recent")
app.add_typer(threshold_use_recent, name="threshold-use-recent")

if __name__ == "__main__":
    app()
