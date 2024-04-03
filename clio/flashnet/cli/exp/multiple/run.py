import typer

from . import admit_confidence, admit_entropy, admit_uncertain

app = typer.Typer()

app.command(name="admit.uncertain")(admit_uncertain.exp_admit_uncertain)
app.command(name="admit.entropy")(admit_entropy.exp_admit_entropy)
app.command(name="admit.confidence")(admit_confidence.exp_admit_confidence)

if __name__ == "__main__":
    app()
