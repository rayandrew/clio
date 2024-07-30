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

def calculate_characteristic(data, log):
    n_data = len(data)
    read_count = int((data["io_type"] == 1).sum())
    write_count = n_data - read_count
    reject_count = int(data["reject"].sum())
    accept_count = n_data - reject_count
    min_ts_record = int(data["ts_record"].min())
    max_ts_record = int(data["ts_record"].max())
    print("MIN TS RECORD", min_ts_record)
    print("MAX TS RECORD", max_ts_record)
    duration = max_ts_record - min_ts_record
    readonly_data = data[data["io_type"] == 1]
    writeonly_data = data[data["io_type"] == 0]
    print("Generating size...")
    size = Statistic.generate(data["size"].values)
    print("Generating read size...")
    read_size = Statistic.generate(readonly_data["size"].values)
    print("Generating write size...")
    write_size = Statistic.generate(writeonly_data["size"].values)
    print("Generating offset...")
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
        reject_count=reject_count,
        accept_count=accept_count,
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
    return characteristic

@app.command()
def characteristic_list_of_window(
    data_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    seed: Annotated[int, typer.Option(help="The seed to use", show_default=True)] = 3003,
):
    args = locals()


    general_set_seed(seed)

    # window_size = parse_time(window_size)

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    characteristic_dir = output 
    # traces = parse_list_file(list_of_window)
    data_dir_csv_dict = {}
    
    from natsort import natsorted
    
    for trace_csv in natsorted(data_dir.glob("*.csv")):
        trace_name = trace_csv.stem
        data_dir_csv_dict[trace_name] = trace_csv

    characteristics = Characteristics()
    characteristics_per_window = Characteristics()

    curr_characteristic_dir = characteristic_dir 
    curr_characteristic_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame()
    
    from clio.flashnet.preprocessing.labeling import labeling
    for i, (trace_name, trace_csv) in enumerate(data_dir_csv_dict.items()):
        if (i % 1000) == 0 and i != 0:
            print(trace_name, trace_csv)
            characteristics_per_window.to_dataframe().to_csv(curr_characteristic_dir / f"per_window_characteristics_{i}.csv")

        data_temp = pd.read_csv(trace_csv, names=["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "size_after_replay"], header=None)
        
        data_temp = normalize_df_ts_record(data_temp, col="ts_record")
        data_temp = normalize_df_ts_record(data_temp, col="ts_submit")
        data_temp = labeling(data_temp)
        
        ts_offset = i * 60 * 1000
        data_temp["ts_record"] += ts_offset
        data_temp["ts_submit"] += ts_offset
        char = calculate_characteristic(data_temp, log)
        characteristics_per_window.append(char)
        data = pd.concat([data, data_temp], ignore_index=True)
    # data.to_csv(curr_characteristic_dir / "data_full.csv")
        
    characteristic = calculate_characteristic(data, log)
    
    characteristics.append(characteristic)
    characteristics.to_dataframe().to_csv(curr_characteristic_dir / "full_characteristics.csv")
    characteristics_per_window.to_dataframe().to_csv(curr_characteristic_dir / "per_window_characteristics.csv")
    # make a done file
    (curr_characteristic_dir / "done").touch()
    
    return

if __name__ == "__main__":
    app()
    ## python -m clio.flashnet.cli.characteristic list_of_window \ "data/flashnet/characteristics/calculate/1m/alibaba/" \--list-file data/flashnet/characteristics/calculate/1m/alibaba/list_of_window.txt \--output data/flashnet/characteristics/list_of_window/1m/alibaba
