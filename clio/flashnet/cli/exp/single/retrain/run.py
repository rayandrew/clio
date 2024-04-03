import typer

from . import all_data, confidence_based, entropy_based, uncertain_based

app = typer.Typer()
app.command(name="window")(all_data.exp_all_window_data)
app.command(name="uncertain")(uncertain_based.exp_uncertain_based)
app.command(name="entropy")(entropy_based.exp_entropy_based)
app.command(name="confidence")(confidence_based.exp_confidence_based)

if __name__ == "__main__":
    app()
