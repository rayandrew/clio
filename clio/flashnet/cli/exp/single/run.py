import typer

from . import initial_only
from .renew.run import app as renew
from .retrain.run import app as retrain

app = typer.Typer()
app.command(name="initial-only")(initial_only.exp_initial_only)
app.add_typer(renew, name="renew")
app.add_typer(retrain, name="retrain")


if __name__ == "__main__":
    app()
