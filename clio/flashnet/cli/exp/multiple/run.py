import typer

from . import admit_entropy, admit_uncertain, matchmaker

app = typer.Typer()

app.command(name="admit.uncertain")(admit_uncertain.exp_admit_uncertain)
app.command(name="admit.entropy")(admit_entropy.exp_admit_entropy)
app.command(name="matchmaker.batch")(matchmaker.exp_matchmaker_batch)
app.command(name="matchmaker.window")(matchmaker.exp_matchmaker_all)
app.command(name="matchmaker.single")(matchmaker.exp_matchmaker_batch)
if __name__ == "__main__":
    app()
