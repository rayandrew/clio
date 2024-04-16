import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import typer

from clio.utils.general import general_set_seed
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import default_timer

app = typer.Typer(name="Trace Characteristics -- Compare Average Median", pretty_exceptions_enable=False)


@app.command()
def compare_average_median(
    generate_list_file: Annotated[Path, typer.Argument(help="The generate list file", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    characteristic_file: Annotated[Path, typer.Argument(help="The characteristic file", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: LogLevel = LogLevel.INFO,
    seed: int = 3003,
):
    args = locals()

    global_start_time = default_timer()

    general_set_seed(seed)

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    # Read characteristic file .csv
    characteristic = pd.read_csv(characteristic_file)

    traces: dict[str, dict[str, str]] = {}
    key = None
    counter = 0
    with open(generate_list_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                continue

            # remove # .... from line
            line = line.split("#")[0]
            line = line.strip()

            if line.startswith("!"):
                counter = 0
                key = line[1:]
                key = key.strip()
                if key not in traces:
                    traces[key] = {}
                continue

            if ":" in line:
                name, value = line.split(":")
                name = name.strip()
                value = value.strip()
                traces[key][name] = value
                continue

            if key is not None:
                traces[key][counter] = line
                counter += 1

    for key, trace_dict in traces.items():
        # join the multiplier on name based on trace_dict keys
        filtered_characteristic = characteristic.merge(pd.DataFrame(trace_dict.items(), columns=["multiplier", "name"]), on="name")

        identifier = key.split("_")[:-1]
        # join the list into str
        identifier = "_".join(identifier)
        y_col1 = identifier + "_avg"
        y_col2 = identifier + "_median"
        keep_column = ["name", "multiplier", y_col1, y_col2]
        filtered_characteristic = filtered_characteristic[keep_column]

        # make a new plot will be a side by side barchart
        # X is multiplier, will be discrete.
        # Y is y_col1 and y_col2, will each have their own barchart side by side

        filtered_characteristic = filtered_characteristic.sort_values(by="multiplier")
        print(filtered_characteristic)

        melted_df = filtered_characteristic.melt(id_vars="multiplier", var_name="variable", value_name="value")

        # drop rows with value "name" in variable
        melted_df = melted_df[melted_df["variable"] != "name"]

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(12, 5))

        print(melted_df)

        # Plot the bar chart
        sns.barplot(x="multiplier", y="value", hue="variable", data=melted_df, palette="tab10")

        # Set title and labels
        ax.set_title(identifier.capitalize() + " Average vs Median")
        ax.set_xlabel("Multiplier")
        ax.set_ylabel("Value")

        # Adjust layout
        fig.tight_layout()

        # Save the plot
        output_path = output / (str(identifier) + "_avg_vs_median.png")
        plt.savefig(output_path, dpi=300)

        # Close the plot
        plt.close(fig)

        # Logging
        log.info("Saved plot to %s", output_path)

        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.set_title(f"{identifier} Average vs Median")
        # ax.set_xlabel("Multiplier")
        # ax.set_ylabel("Value")
        # sns.barplot(x="multiplier", y=y_col1, data=filtered_characteristic, ax=ax, hue="multiplier", palette="tab10")
        # sns.barplot(x="multiplier", y=y_col2, data=filtered_characteristic, ax=ax, hue="multiplier", palette="tab10")
        # fig.tight_layout()
        # fig.savefig(output / f"{identifier}_avg_vs_median.png", dpi=300)
        # plt.close(fig)
        # log.info("Saved plot to %s", output / f"{identifier}_avg_vs_median.png", tab=0)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)
