from enum import Enum
from pathlib import Path
from typing import Annotated
from pandas import read_csv

import typer

from clio.utils.csv import read_csv_gz
from clio.utils.logging import log_global_setup
from clio.utils.trace import TraceEntry, TraceWriter

app = typer.Typer(name="Standardizer")


@app.command()
def msrc(
    files: Annotated[list[Path], "The file(s) to analyze"],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
):
    """
    Standardize the MSRC trace format to MSFT format.

    :param files (list[Path]): The file(s) to analyze
    :param output (Path): The output path to write the results to
    """
    assert len(files) > 0, "No files to analyze"

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", console_width=None)
    min_time = 0

    writer = TraceWriter(
        output / f"{output.stem}.trace",
    )
    for i, file in enumerate(files):
        log.info("Standardizing %s", file, tab=0)
        for j, (row, _) in enumerate(read_csv_gz(file, contains_header=False)):
            if i == 0 and j == 0:
                min_time = float(row[0])

            entry = TraceEntry(
                ts_record=(float(row[0]) - min_time) * 0.00001,
                disk_id=row[2],
                offset=int(row[4]),
                io_size=int(row[5]),
                read=row[3] == "Read",
            )
            writer.write(entry)
    writer.close()


@app.command()
def tectonic(
    files: Annotated[list[Path], "The file(s) to analyze"],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
):
    """
    Standardize the Tectonic trace format to MSFT format.

    :param files (list[Path]): The file(s) to analyze
    :param output (Path): The output path to write the results to
    """
    assert len(files) > 0, "No files to analyze"
    
    filename = output.stem
    output = output.parent
    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", console_width=None)
    writer = TraceWriter(
        output / f"{filename}.trace",
    )
    
    file_raw = open(output/ f"{filename}_raw.trace", "w")
    
    ## Taken from Baleen's codebase
    ## https://github.com/wonglkd/BCacheSim/blob/ddeb2d8035483b5943fa57df1932ffc7d1134b6d/cachesim/legacy_utils.py#L72
    MAX_BLOCK_SIZE = 8 * 1024 * 1024
    class OpType(Enum):
        GET_TEMP = 1
        GET_PERM = 2
        PUT_TEMP = 3
        PUT_PERM = 4
        GET_NOT_INIT = 5
        PUT_NOT_INIT = 6
        UNKNOWN = 100

    PUT_OPS = [OpType.PUT_PERM.value, OpType.PUT_TEMP.value, OpType.PUT_NOT_INIT.value]
    GET_OPS = [OpType.GET_PERM.value, OpType.GET_TEMP.value, OpType.GET_NOT_INIT.value]
    
    for i, file in enumerate(files):
        log.info("Standardizing %s", file, tab=0)
        ## if filename contains "2021", then process
        if "2021" in str(file):
            ## block_id io_offset io_size op_time op_name user_namespace user_name host_name op_count
            MAP_DICT = {
                "block_id": 0,
                "io_offset": 1,
                "io_size": 2,
                "op_time": 3,
                "op_name": 4,
                "user_namespace": 5,
                "user_name": 6,
                "host_name": 7,
                "op_count": 8,
            }
            
        elif "2019" in str(file):
            ## block_id io_offset io_size op_time op_name pipeline user_namespace user_name
            MAP_DICT = {
                "block_id": 0,
                "io_offset": 1,
                "io_size": 2,
                "op_time": 3,
                "op_name": 4,
                "pipeline": 5,
                "user_namespace": 6,
                "user_name": 7,
            }
            
            log.info("Year 2019 has no host id! Substituting with usernamespace for now..")
        elif "2023" in str(file):
            ## block_id io_offset io_size op_time op_name user_namespace user_name rs_shard_id op_count host_name
            MAP_DICT = {
                "block_id": 0,
                "io_offset": 1,
                "io_size": 2,
                "op_time": 3,
                "op_name": 4,
                "user_namespace": 5,
                "user_name": 6,
                "rs_shard_id": 7,
                "op_count": 8,
                "host_name": 9,
            }
        else:
            log.info("Skipping %s", file, tab=1)
            continue
        
        file_to_read = read_csv(file, header=2, sep=" ", names=MAP_DICT.keys())
        file_to_read.to_csv(file_raw, index=False)
        min_time = file_to_read["op_time"].min()
        # iteratoe through file to read
        num_rows = file_to_read.shape[0]
        for idx, row in file_to_read.iterrows():
            if idx % 100000 == 0:
                log.info("Processed %s/%s rows (%s percnt.)", idx, num_rows, float(idx)/num_rows, tab=1)
            
            block_offset = int(row[MAP_DICT["block_id"]]) * MAX_BLOCK_SIZE
            disk_key = MAP_DICT["host_name"] if "host_name" in MAP_DICT else MAP_DICT["user_namespace"]
            
            entry = TraceEntry(
                ts_record=row[MAP_DICT["op_time"]]-min_time,
                disk_id=row[disk_key],
                offset=block_offset + row[MAP_DICT["io_offset"]],
                io_size=int(row[MAP_DICT["io_size"]]),
                read=int(row[MAP_DICT["op_name"]]) in GET_OPS,
            )
            writer.write(entry)
    writer.close()
    file_raw.close()

@app.command(name="analyze")
def analyze(): ...


if __name__ == "__main__":
    app()
