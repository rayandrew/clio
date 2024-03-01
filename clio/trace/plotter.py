import sys
from pathlib import Path
from typing import Annotated

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from clio.utils.characteristic import Characteristic, Characteristics, Statistic
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.query import QueryExecutionException, get_query

app = typer.Typer(name="Plotter", pretty_exceptions_enable=False)


def mult_normalize(df: pd.DataFrame, exclude: list[str] = []) -> pd.DataFrame:
    # normalize columns by treating min as 1 and other values are values / min
    norm_df = df.copy()
    cols = [col for col in df.columns if col not in exclude]
    for column in cols:
        norm_df[column] = norm_df[column] / norm_df[column].min()
    return norm_df


@app.command()
def characteristics(
    file: Annotated[Path, typer.Argument(help="The characteristics file to plot", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    query: Annotated[str, typer.Option(help="The query to filter the data")] = "",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    """
    Plot the results of the analysis

    :param file (Path): The stats file to plot
    :param output (Path): The output path to write the results to
    :param query (str): The query to filter the data
    :param log_level (LogLevel): The log level to use
    """

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)
    log.info("Plotting %s", file, tab=0)

    characteristics = Characteristics.from_msgpack(file)

    try:
        q = get_query(query)
        new_characteristics = characteristics.query(lambda x: q(x.to_dict())) if q else characteristics
        characteristics = new_characteristics
    except QueryExecutionException as e:
        log.error("Failed to execute expression: %s", e)
        log.info("Skipping the query", tab=1)

    log.info("Processing %d characteristics", len(characteristics), tab=1)

    data = characteristics.to_dataframe()
    data.to_csv(output / "characteristics.csv", index=False)
    data = mult_normalize(data, exclude=["start_ts", "end_ts", "duration", "read_count", "write_count", "ts_unit", "size_unit", "disks"])
    data.to_csv(output / "norm_mult_characteristics.csv", index=False)

    # data["x"] = np.arange(len(data)) + 1

    plot_path = output / "p"

    COLORS = {
        "iops": "slateblue",
        "size_avg": "crimson",
        "iat_avg": "#0f5b2b",
        "rw_ratio": "dodgerblue",
        "num_io": "darkorange",
    }

    N_DATA_PER_PLOT = 30

    bar_plot_path = output / "bar"
    bar_plot_path.mkdir(parents=True, exist_ok=True)

    area_plot_path = output / "area"
    area_plot_path.mkdir(parents=True, exist_ok=True)

    # split the data into (len(data) / N_DATA_PER_PLOT) + 1 chunks
    # and plot each chunk
    last_x = 0
    for i, stride in enumerate(range(0, len(data), N_DATA_PER_PLOT)):
        ### Bar plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.set_title("Characteristics")
        bottom: int = 0
        bar_width = 0.5
        chunk = data.iloc[stride : stride + N_DATA_PER_PLOT]
        chunk["x"] = last_x + np.arange(len(chunk))
        for col in COLORS.keys():
            ax.bar(chunk["x"], chunk[col], bar_width, bottom=bottom, label=col, color=COLORS[col])
            bottom += chunk[col]

        for bar in ax.patches:
            bar.set_edgecolor("black")
            bar.set_linewidth(0.5)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                (bar.get_height() / 2) + bar.get_y(),
                f"{round(bar.get_height(), 2)}x",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Window")
        ax.set_ylabel("Multiplier")
        ax.set_xlim(last_x - bar_width, last_x + len(chunk) - bar_width)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=False, shadow=False, frameon=False)
        fig.tight_layout()
        fig.savefig(bar_plot_path / f"characteristics_{i}.png", dpi=300)
        fig.savefig(bar_plot_path / f"characteristics_{i}.eps", dpi=300)
        plt.close(fig)

        ### Area plot

        # log.info("X: %s", chunk["x"].values)

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.set_title("Characteristics")

        # chunk["iops"] += chunk["num_io"]
        # chunk["size_avg"] += chunk["iops"]
        # chunk["iat_avg"] += chunk["size_avg"]
        # chunk["rw_ratio"] += chunk["iat_avg"]

        ax.stackplot(
            chunk["x"],
            chunk["num_io"],
            chunk["iops"],
            chunk["size_avg"],
            chunk["iat_avg"],
            chunk["rw_ratio"],
            labels=list(COLORS.keys()),
            colors=list(COLORS.values()),
        )
        ax.set_xlabel("Window")
        ax.set_ylabel("Multiplier")
        ax.set_xlim(last_x, last_x + len(chunk) - 1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=False, shadow=False, frameon=False)
        fig.tight_layout()
        fig.savefig(area_plot_path / f"characteristics_{i}.png", dpi=300)
        fig.savefig(area_plot_path / f"characteristics_{i}.eps", dpi=300)
        plt.close(fig)

        last_x = chunk["x"].max() + 1

    # for col in ["num_io", "iops", "size_avg", "iat_avg", "rw_ratio"]:
    #     ax.bar(data["x"], data[col], bar_width, bottom=bottom, label=col, color=COLORS[col])
    #     bottom += data[col]
    #
    # for bar in ax.patches:
    #     bar.set_edgecolor("black")
    #     bar.set_linewidth(0.5)
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         (bar.get_height() / 2) + bar.get_y(),
    #         f"{round(bar.get_height(), 2)}x",
    #         ha="center",
    #         va="center",
    #         color="white",
    #         fontsize=10,
    #         fontweight="bold",
    #     )
    #
    # # remove y-axis
    # ax.yaxis.set_ticks([])
    #
    # ax.set_xlabel("Window")
    # ax.set_ylabel("Multiplier")
    #
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=False, shadow=False, frameon=False)
    #
    # fig.tight_layout()
    #
    # fig.savefig(output / "characteristics.png", dpi=300)
    #


@app.command()
def temp(): ...


if __name__ == "__main__":
    app()
