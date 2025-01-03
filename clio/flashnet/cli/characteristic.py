import typer

from clio.flashnet.cli.characteristic_ import (
    analyze,
    calculate,
    characteristic_list_of_window,
    compare_average_median,
    generate,
    list_generator,
    msft,
    revert_to_replay,
    select_data,
    split,
    concept_finder,
)

app = typer.Typer(name="Trace Characteristics", pretty_exceptions_enable=False)
app.command(name="revert-to-replay")(revert_to_replay.revert_to_replay)
app.command(name="split")(split.split)
app.command(name="generate")(generate.generate)
app.command(name="listgenerator")(list_generator.list_generator)
app.command(name="driftlistgenerator")(list_generator.list_generator_drift)
app.command(name="characteristic")(characteristic_list_of_window.characteristic_list_of_window)
app.command(name="selectdata")(select_data.select_data)
app.command(name="calculate")(calculate.calculate)
app.command(name="analyze")(analyze.analyze)
app.command(name="compare-average-median")(compare_average_median.compare_average_median)
app.command(name="msft")(msft.analyze)
app.command(name="generate_v2")(generate.generate_v2)
app.command(name="concept_finder")(concept_finder.concept_finder)
app.command(name="rescale")(generate.rescale)
app.command(name="generate_plot_range")(analyze.generate_plot_range)
app.command(name="generate_plot_spider")(analyze.generate_plot_spider)

if __name__ == "__main__":
    app()
    ## python -m clio.flashnet.cli.characteristic list-generator \ "data/flashnet/characteristics/calculate/1m/alibaba/" \--output data/flashnet/characteristics/generate_list/1m/alibaba
