import typer

from . import initial_only, initial_only_with_train_data
from .renew.run import app as renew
from .retrain.run import app as retrain

app = typer.Typer()
app.command(name="initial-only")(initial_only.exp_initial_only)
app.command(name="initial-only-with-train-data")(initial_only_with_train_data.exp_initial_only_with_train_data)
app.add_typer(renew, name="renew")
app.add_typer(retrain, name="retrain")

if __name__ == "__main__":
    app()
