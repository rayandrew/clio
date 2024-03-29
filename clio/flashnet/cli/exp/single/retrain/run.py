import typer

from . import all_data, entropy_based, uncertainty_based

app = typer.Typer()
app.command(name="all-data")(all_data.exp_all_data)
app.command(name="uncertainty")(uncertainty_based.exp_uncertainty_based)
app.command(name="entropy")(entropy_based.exp_entropy_based)

if __name__ == "__main__":
    app()
