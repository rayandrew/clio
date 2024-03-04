import io
import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
from sklearn.preprocessing import StandardScaler

import typer

from clio.utils.characteristic import Characteristic, Characteristics, Statistic
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.query import QueryExecutionException, get_query
from clio.utils.trace_pd import TraceWindowGeneratorContext, read_dataset_as_df, read_labeled_as_df, trace_time_window_generator

app = typer.Typer(name="Characteristic Clustering", pretty_exceptions_enable=False)


def preliminary_process(output: Path, log_level: LogLevel, dir: Path, query: str):
    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Run clustering on %s", dir, tab=0)

    is_dataset = True
    data_paths = [p for p in dir.glob("**/*.dataset")]
    if len(data_paths) == 0:
        data_paths = [p for p in dir.glob("**/*.labeled")]
        is_dataset = False

    if len(data_paths) == 0:
        log.error("No datasets found in %s", dir)
        sys.exit(1)

    reader = read_dataset_as_df if is_dataset else read_labeled_as_df

    # TODO: remove this
    # for debugging purposes, only use the first dataset
    data_paths = [data_paths[0]]

    try:
        q = get_query(query)
        data = pd.concat(reader(path) for path in data_paths)
        if q:
            data: pd.DataFrame = data[q({"data": data})]  # type: ignore
    except QueryExecutionException as e:
        log.error("Failed to execute expression: %s", e)
        sys.exit(1)

    scaler = StandardScaler()
    # remove latency and reject if any
    data = data.drop(columns=["latency", "reject"], errors="ignore")
    data = data.dropna()
    log.info("Data columns: %s", list(data.columns), tab=1)

    return log, data


@app.command()
def dbscan(
    dir: Annotated[Path, typer.Argument(help="The characteristics file to plot", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    query: Annotated[str, typer.Option(help="The query to filter the data")] = "",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    from sklearn.cluster import DBSCAN

    log, data = preliminary_process(output, log_level, dir, query)

    dbscan = DBSCAN(eps=0.2, min_samples=500, n_jobs=-1)
    dbscan = dbscan.fit(data)

    labels = set(dbscan.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # type: ignore
    log.info("Number of clusters: %d", n_clusters, tab=1)
    n_noise_ = list(labels).count(-1)  # type: ignore
    log.info("Number of noise points: %d", n_noise_, tab=1)


@app.command()
def hdbscan(
    dir: Annotated[Path, typer.Argument(help="The characteristics file to plot", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    query: Annotated[str, typer.Option(help="The query to filter the data")] = "",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    from sklearn.cluster import HDBSCAN

    log, data = preliminary_process(output, log_level, dir, query)

    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=10)
    hdbscan = hdbscan.fit(data)

    labels = set(hdbscan.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # type: ignore
    log.info("Number of clusters: %d", n_clusters, tab=1)
    n_noise_ = list(labels).count(-1)  # type: ignore
    log.info("Number of noise points: %d", n_noise_, tab=1)


if __name__ == "__main__":
    app()
