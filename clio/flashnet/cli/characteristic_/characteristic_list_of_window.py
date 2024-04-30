import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import typer

from clio.flashnet.cli.characteristic_.utils import parse_list_file

from clio.utils.characteristic import Characteristic, Characteristics, Statistic
from clio.utils.general import general_set_seed, parse_time
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import default_timer
from clio.utils.trace_pd import normalize_df_ts_record

app = typer.Typer(name="Trace Characteristics -- Characteristic List of Window", pretty_exceptions_enable=False)


@app.command()
def characteristic_list_of_window(
    data_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    list_of_window: Annotated[
        Path, typer.Option("--list-file", help="The list of window files", exists=True, file_okay=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "1m",
    # feat_name: Annotated[str, typer.Option(help="The feature name", show_default=True)] = "feat_v6_ts",
    # window_agg_size: Annotated[int, typer.Option(help="The window aggregation size (in number of I/Os)", show_default=True)] = 10,
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    seed: Annotated[int, typer.Option(help="The seed to use", show_default=True)] = 3003,
):
    args = locals()

    global_start_time = default_timer()

    general_set_seed(seed)

    window_size = parse_time(window_size)

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    characteristic_dir = output / "characteristic"
    traces = parse_list_file(list_of_window)

    characteristics = Characteristics()

    for trace_group, trace_dict in traces.items():
        log.info("Trace group: %s", trace_group, tab=0)

        curr_characteristic_dir = characteristic_dir / trace_group
        curr_characteristic_dir.mkdir(parents=True, exist_ok=True)

        for i, (trace_name, trace) in enumerate(trace_dict.items()):
            p = data_dir / f"{trace}.csv"
            data = pd.read_csv(p)
            data = normalize_df_ts_record(data)
            data = data.drop(columns=["ts_submit", "original_ts_record", "size_after_replay", "latency", "size_after_replay", "reject"], errors="ignore")

            n_data = len(data)
            read_count = int((data["io_type"] == 1).sum())
            write_count = n_data - read_count
            min_ts_record = int(data["ts_record"].min())
            max_ts_record = int(data["ts_record"].max())
            duration = max_ts_record - min_ts_record
            readonly_data = data[data["io_type"] == 1]
            writeonly_data = data[data["io_type"] == 0]
            log.debug("Generating size...")
            size = Statistic.generate(data["size"].values)
            log.debug("Generating read size...")
            read_size = Statistic.generate(readonly_data["size"].values)
            log.debug("Generating write size...")
            write_size = Statistic.generate(writeonly_data["size"].values)
            log.debug("Generating offset...")
            offset = Statistic.generate(data["offset"].values)
            log.debug("Generating iat...")
            iat = data["ts_record"].diff().dropna()
            iat[iat < 0] = 0
            iat = Statistic.generate(iat.values)
            read_iat = readonly_data["ts_record"].diff().dropna()
            read_iat[read_iat < 0] = 0
            read_iat = Statistic.generate(read_iat.values)
            write_iat = writeonly_data["ts_record"].diff().dropna()
            write_iat[write_iat < 0] = 0
            write_iat = Statistic.generate(write_iat.values)
            if "latency" in data.columns:
                log.debug("Generating throughput...")
                throughput = Statistic.generate((data["size"] / data["latency"]).values)
                read_throughput = Statistic.generate((readonly_data["size"] / readonly_data["latency"]).values)
                write_throughput = Statistic.generate((writeonly_data["size"] / writeonly_data["latency"]).values)
                log.debug("Generating latency...")
                latency = Statistic.generate(data["latency"].values)
                read_latency = Statistic.generate(readonly_data["latency"].values)
                write_latency = Statistic.generate(writeonly_data["latency"].values)
            else:
                log.debug("Generating fake throughput...")
                latency = np.zeros(len(data))
                latency -= 1.0
                throughput = Statistic.generate(data["size"].values / latency)
                latency = Statistic.generate(latency)
                read_latency = np.zeros(len(readonly_data))
                read_latency -= 1.0
                read_throughput = Statistic.generate(readonly_data["size"].values / read_latency)
                read_latency = Statistic.generate(read_latency)
                write_latency = np.zeros(len(writeonly_data))
                write_latency -= 1.0
                write_throughput = Statistic.generate(writeonly_data["size"].values / write_latency)
                write_latency = Statistic.generate(write_latency)
            characteristic = Characteristic(
                num_io=n_data,
                disks=set([0]),
                start_ts=min_ts_record,
                end_ts=max_ts_record,
                duration=duration,
                ts_unit="ms",
                read_count=read_count,
                write_count=write_count,
                size=size,
                read_size=read_size,
                write_size=write_size,
                offset=offset,
                iat=iat,
                read_iat=read_iat,
                write_iat=write_iat,
                throughput=throughput,
                read_throughput=read_throughput,
                write_throughput=write_throughput,
                latency=latency,
                read_latency=read_latency,
                write_latency=write_latency,
            )
            characteristics.append(characteristic)

        characteristics.to_dataframe().to_csv(curr_characteristic_dir / "characteristics.csv")

        mult_characteristics = characteristics.generate_multipliers()
        mult_characteristics_df = mult_characteristics.to_dataframe()

        # plot characteristics in stacked bar plot
        # plot num_io, read_count, write_count, size, read_size, write_size, offset, iat, read_iat, write_iat, throughput, read_throughput, write_throughput, latency, read_latency, write_latency
        # for each multiplier
        for col in mult_characteristics_df.columns:
            if col in ["disks", "start_ts", "end_ts", "ts_unit", "size_unit", "duration", "num_disks"]:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            mult_characteristics_df.plot(kind="bar", y=col, ax=ax)
            ax.set_title(f"{col}")
            ax.set_xlabel("Window")
            ax.set_ylabel(col)
            fig.tight_layout()
            fig.savefig(curr_characteristic_dir / f"characteristics_{col}.png", dpi=300)
            plt.close(fig)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
    ## python -m clio.flashnet.cli.characteristic list_of_window \ "data/flashnet/characteristics/calculate/1m/alibaba/" \--list-file data/flashnet/characteristics/calculate/1m/alibaba/list_of_window.txt \--output data/flashnet/characteristics/list_of_window/1m/alibaba
