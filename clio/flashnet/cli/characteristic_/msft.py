import json
import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import typer

from clio.utils.characteristic import Characteristic, CharacteristicDict, Statistic
from clio.utils.general import parse_time
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import default_timer

app = typer.Typer(name="Trace Characteristics -- MSFT", pretty_exceptions_enable=False)


def _set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


@app.command()
def analyze(
    data_path: Annotated[Path, typer.Argument(help="The data path to use", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    data = pd.read_csv(data_path, header=None, names=["ts_record", "disk_id", "offset", "size", "io_type"], delimiter=" ")
    data = data.sort_values(by="ts_record")

    # io_type == 1 -> read
    # io_type == 0 -> write
    min_ts_record = data["ts_record"].min()
    max_ts_record = data["ts_record"].max()
    n_data = len(data)
    duration = max_ts_record - min_ts_record
    readonly_data = data[data["io_type"] == 1]
    read_count = len(readonly_data)
    writeonly_data = data[data["io_type"] == 0]
    write_count = len(writeonly_data)
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
    log.debug("Generating fake latency...")
    latency = np.zeros(len(data))
    latency -= 1.0
    throughput = Statistic.generate(data["size"].values / latency)
    latency = Statistic.generate(latency)
    fake_read_latency = np.zeros(len(readonly_data))
    fake_read_latency -= 1.0
    read_latency = Statistic.generate(fake_read_latency)
    fake_write_latency = np.zeros(len(writeonly_data))
    fake_write_latency -= 1.0
    write_latency = Statistic.generate(fake_write_latency)
    log.debug("Generating fake throughput...")
    read_throughput = Statistic.generate(readonly_data["size"].values / fake_read_latency)
    write_throughput = Statistic.generate(writeonly_data["size"].values / fake_write_latency)
    characteristic = Characteristic(
        num_io=n_data,
        disks=set(["0"]),
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
    characteristic.to_msgpack(output / "characteristics.msgpack")
    with open(output / "characteristics.json", "w") as f:
        json.dump(characteristic.to_dict(), f, indent=4, default=_set_default)

    with open(output / "characteristics_flat.json", "w") as f:
        json.dump(characteristic.to_flat_dict(), f, indent=4, default=_set_default)

    with IndentedFile(output / "characteristics.txt") as ifile:
        characteristic.to_indented_file(ifile)

    global_end_time = default_timer()
    log.info("Total time: %.4f sec", global_end_time - global_start_time)


if __name__ == "__main__":
    app()
