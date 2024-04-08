import typer

from . import admit_confidence, admit_entropy, admit_uncertain, admit_window, matchmaker, aue

app = typer.Typer()

app.command(name="admit.window")(admit_window.exp_admit_window)
app.command(name="admit.uncertain")(admit_uncertain.exp_admit_uncertain)
app.command(name="admit.entropy")(admit_entropy.exp_admit_entropy)
app.command(name="admit.confidence")(admit_confidence.exp_admit_confidence)
app.command(name="matchmaker.batch")(matchmaker.exp_matchmaker_batch)
app.command(name="matchmaker.window")(matchmaker.exp_matchmaker_all)
app.command(name="matchmaker.single")(matchmaker.exp_matchmaker_batch)
app.command(name="aue.scikit")(aue.exp_aue)
app.command(name="aue.flashnet")(aue.exp_aue_adapted)

if __name__ == "__main__":
    app()
