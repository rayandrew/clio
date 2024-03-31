import typer

from . import all_data, confidence_based, entropy_based, uncertainty_based

app = typer.Typer()
app.command(name="all-data")(all_data.exp_all_data)
app.command(name="uncertainty")(uncertainty_based.exp_uncertainty_based)
app.command(name="entropy")(entropy_based.exp_entropy_based)
app.command(name="confidence")(confidence_based.exp_confidence_based)

if __name__ == "__main__":
    app()
