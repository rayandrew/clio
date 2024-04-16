import typer

from clio.flashnet.cli.characteristic import (
    analyze,
    calculate,
    characteristic_list_of_window,
    compare_average_median,
    generate,
    list_generator,
    revert_to_replay,
    select_data,
    split,
)

app = typer.Typer(name="Trace Characteristics", pretty_exceptions_enable=False)
app.command(name="revert-to-heimdall-replay-data")(revert_to_replay.heimdall)
app.command(name="revert-to-linnos-replay-data")(revert_to_replay.linnos)
app.command(name="replay-heimdall-to-linnos")(revert_to_replay.heimdall_to_linnos)
app.command(name="replay-linnos-to-heimdall")(revert_to_replay.linnos_to_heimdall)
app.command(name="split")(split.split)
app.command(name="generate")(generate.generate)
app.command(name="listgenerator")(list_generator.listgenerator)
app.command(name="characteristic")(characteristic_list_of_window.characteristic_list_of_window)
app.command(name="selectdata")(select_data.select_data)
app.command(name="calculate")(calculate.calculate)
app.command(name="analyze")(analyze.analyze)
app.command(name="compare-average-median")(compare_average_median.compare_average_median)


if __name__ == "__main__":
    app()
    ## python -m clio.flashnet.cli.characteristic list-generator \ "data/flashnet/characteristics/calculate/1m/alibaba/" \--output data/flashnet/characteristics/generate_list/1m/alibaba
