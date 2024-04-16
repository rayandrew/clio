import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import typer

from clio.flashnet.cli.characteristic.utils import mult_normalize

from clio.utils.characteristic import CharacteristicDict
from clio.utils.general import general_set_seed, parse_time
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import default_timer

app = typer.Typer(name="Trace Characteristics -- Select Data", pretty_exceptions_enable=False)


@app.command()
def select_data(
    data_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    characteristic_path: Annotated[
        Path, typer.Option("--characteristic", help="The characteristic file", exists=True, file_okay=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "1m",
    # feat_name: Annotated[str, typer.Option(help="The feature name", show_default=True)] = "feat_v6_ts",
    # window_agg_size: Annotated[int, typer.Option(help="The window aggregation size (in number of I/Os)", show_default=True)] = 10,
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    seed: Annotated[int, typer.Option(help="The seed to use", show_default=True)] = 3003,
    duration: Annotated[str, typer.Option(help="The duration to use (in minute(s))", show_default=True)] = "1h",
    num_variations: Annotated[int, typer.Option(help="The number of variations to use", show_default=True)] = 5,
):
    args = locals()

    global_start_time = default_timer()

    general_set_seed(seed)

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    ##########################################################################
    # Data selection
    ##########################################################################

    window_size = parse_time(window_size)  # minute
    duration = parse_time(duration)  # minute
    window_count = duration // window_size

    log.info("Window size: %s", window_size, tab=0)
    log.info("Duration: %s", duration, tab=0)
    log.info("Window count: %s", window_count, tab=0)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)

    characteristic = CharacteristicDict.from_msgpack(characteristic_path)
    log.info("Loaded characteristics from %s", characteristic_path, tab=0)

    characteristics_df = characteristic.to_dataframe()
    log.info("Characteristics dataframe shape: %s", characteristics_df.shape, tab=0)

    mult_characteristics_df = mult_normalize(characteristics_df)

    # select data based on the characteristics with duration
    # find random drift with the variation of
    # - read_size_avg
    # - read_latency_avg
    # - read_iat_avg
    # - read_throughput_avg
    # - size_avg
    # - latency_avg
    # - iat_avg
    # - throughput_avg
    # - write_size_avg
    # - write_latency_avg
    # - write_iat_avg
    # - write_throughput_avg
    # - num_io

    # find maximum 5 variations of each characteristic
    CHARACTERISTIC_CRITERIA = [
        "read_size_avg",
        "read_latency_avg",
        "read_iat_avg",
        "read_throughput_avg",
        "size_avg",
        "latency_avg",
        "iat_avg",
        "throughput_avg",
        "write_size_avg",
        "write_latency_avg",
        "write_iat_avg",
        "write_throughput_avg",
        "num_io",
    ]

    # paths: list[list[Path]] = [[] * len(CHARACTERISTIC_CRITERIA)]
    # names: set[str] = set()
    criterias: dict[str, dict[int, pd.DataFrame]] = {}

    for column in CHARACTERISTIC_CRITERIA:
        char_df = mult_characteristics_df.copy()
        base_df = char_df[char_df[column] == 1]

        mult_dict: dict[int, pd.DataFrame] = {
            1: base_df,
        }
        for mult in [1.2, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            # find the window that has roughly equal to size * mult
            mult_df = char_df[np.isclose(char_df[column], mult, atol=0.1, rtol=0.0)]
            if mult_df.empty:
                # log.info("No window found for %s_mult_%s", column, mult, tab=1)
                continue
            mult_dict[mult] = mult_df  # .sample(n=num_variations) if len(mult_df) > num_variations else mult_df
            # names.update(mult_df["name"])

        # check if the mult_dict contains only the base
        if len(mult_dict) == 1:
            continue

        criterias[column] = mult_dict

        # if column not in criterias:
        #     criterias[column] = []

        # select the data
        # for i in range(num_variations):
        #     # pick random mult
        #     for mult, mult_df in mult_dict.items():
        #         if mult == 1:
        #             continue
        #         # pick random window
        #         window = mult_df.sample(n=1)
        #         criterias[column].append(window["name"].values[0])

    variations: list[list[str]] = []
    log.info("duration: %s", duration, tab=0)
    log.info("variations: %s", variations, tab=0)

    curr_idx = 0

    for i in range(num_variations):
        selected_variations = []
        for column, v in criterias.items():
            for mult, v2 in v.items():
                selected_variations.append(v2["name"].sample(n=1).values[0])

        # select random window_count from selected_variations
        selected_variations = random.sample(selected_variations, window_count)
        # shuffle the selected_variations
        random.shuffle(selected_variations)
        variations.append(selected_variations)

    for i, vari in enumerate(variations):
        with open(output / f"variation_{i}.txt", "w") as f:
            f.write("! var_%s\n" % i)
            for v in vari:
                f.write(v)
                f.write("\n")

    with open(output / "variations.txt", "w") as f:
        for i, vari in enumerate(variations):
            f.write("! var_%s\n" % i)
            for v in vari:
                f.write(v)
                f.write("\n")
            f.write("\n")

    # while curr_idx < num_variations:
    #     while (len(variations[curr_idx]) == 0) or (len(variations[curr_idx]) % (window_count - 1) != 0):
    #         log.info("Current index: %s", curr_idx, tab=0)
    #         for column, v in criterias.items():
    #             for mult, v2 in v.items():
    #                 # for name in v2["name"].values:
    #                 log.info("Mult: %s, Shape: %s", mult, v2.shape, tab=1)
    #                 variations[curr_idx].append(v2["name"].sample(n=1).values[0])
    #     curr_idx += 1

    # randomly pick 5 variations based on mult
    # 1x mult is always picked

    # log.info("variations: %s", len(variations), tab=0)
    # for i, vari in enumerate(variations):
    #     log.info("variations shape: %d %s", i, len(vari), tab=0)

    # log.info("Criterias: %s", criterias, tab=0)


if __name__ == "__main__":
    app()
