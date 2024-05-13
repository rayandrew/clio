import json
import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")


import pandas as pd

import typer

from clio.flashnet.cli.characteristic_.utils import parse_list_file
from clio.flashnet.preprocessing.add_filter import add_filter_v2
from clio.flashnet.preprocessing.feature_engineering import feature_engineering
from clio.flashnet.preprocessing.labeling import labeling
from clio.flashnet.preprocessing.labeling_old import labeling as labeling_old

from clio.utils.general import general_set_seed, parse_time
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import normalize_df_ts_record

app = typer.Typer(name="Trace Characteristics -- Generate", pretty_exceptions_enable=False)


@app.command()
def generate(
    file: Annotated[Path, typer.Argument(help="The trace file", exists=True, file_okay=True, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    device: Annotated[int, typer.Option(help="The device name", show_default=True)] = 0,
    # feat_name: Annotated[str, typer.Option(help="The feature name", show_default=True)] = "feat_v6_ts",
    # window_agg_size: Annotated[int, typer.Option(help="The window aggregation size (in number of I/Os)", show_default=True)] = 10,
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    seed: Annotated[int, typer.Option(help="The seed to use", show_default=True)] = 3003,
):
    args = locals()

    global_start_time = default_timer()

    general_set_seed(seed)

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    df = pd.read_csv(file, names=["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "device", "io_ts"])
    df = df[df["device"] == device]
    df = labeling(df, filter_outlier=False)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)

    # check if


if __name__ == "__main__":
    app()
