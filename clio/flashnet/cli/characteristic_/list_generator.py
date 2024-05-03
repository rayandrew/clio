import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")

import pandas as pd

import typer

from clio.utils.general import general_set_seed
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import default_timer

app = typer.Typer(name="Trace Characteristics -- List Generator", pretty_exceptions_enable=False)


@app.command()
def list_generator(
    data_dir: Annotated[Path, typer.Argument(help="The data directory of calculate", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: LogLevel = LogLevel.INFO,
    num_samples: int = 5,
    seed: int = 3003,
    ## options are sample or pool
    type: Annotated[str, typer.Option(help="Type of list to generate", show_default=True)] = "sample",
):
    ## Input: data_dir, is output of calculate. It will have a /column directory. Each directory in /column/XXX has a bunch of .csv files.
    ## Output: output folder which will have a folder for each XXX in /column. Each folder will have a list of .csv files

    args = locals()

    global_start_time = default_timer()

    general_set_seed(seed)

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    column_dir = data_dir / "column"
    list_of_metrics = list(column_dir.iterdir())

    for metric in list_of_metrics:
        ## get list of csvs in that directory, append to 1 df with a new column for the name of the csv
        log.info("Metric to generate: %s", metric, tab=0)
        df = pd.DataFrame()
        for csv_path in metric.iterdir():
            ## if csv_path is a dir, continue
            if csv_path.is_dir():
                continue
            log.info("CSV: %s", csv_path, tab=1)
            temp_df = pd.read_csv(csv_path)
            temp_df["multiplier"] = csv_path.stem
            # keep columns multiplier, name
            temp_df = temp_df[["multiplier", "name"]]
            df = df._append(temp_df)

        output_dir = output / metric.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        if type == "sample":
            for i in range(num_samples):
                file_name = f"sample_{metric.stem}_{i}.nim"
                df_sample = df.groupby("multiplier").apply(lambda x: x.sample(1))

                # format of file is a .nim file with the following contents
                # ! {file_name}
                # {multiplier}:   {name}

                with open(output_dir / file_name, "w") as f:
                    f.write(f"! {file_name}\n")
                    for index, row in df_sample.iterrows():
                        f.write(f"{row['multiplier']}:   {row['name']}\n")
        else:
            ## make 1 file for each multiplier
            # get multiplier 1
            base_multiplier = df[df["multiplier"] == "1"]
            if base_multiplier.empty:
                log.warning("No base multiplier found", tab=1)
                continue
            base_multiplier = base_multiplier.iloc[0]
            multipliers = df["multiplier"].unique()

            for multiplier_name in multipliers:
                if multiplier_name == "1":
                    continue
                file_name = f"pool_{metric.stem}_{multiplier_name}.nim"
                multiplier_df = df[df["multiplier"] == multiplier_name]

                with open(output_dir / file_name, "w") as f:
                    f.write(f"! {file_name}\n")
                    f.write(f"1:   {base_multiplier['name']}\n")

                    ## sample for 15 from multiplier_df without replacement. If less than 15, just take all
                    if len(multiplier_df) > 15:
                        selected = multiplier_df.sample(n=15, replace=False)
                    else:
                        selected = multiplier_df
                    selected.reset_index(drop=True, inplace=True)

                    for index, row in selected.iterrows():
                        f.write(f"{index+2}:   {row['name']}\n")
                        if index >= 15:
                            break

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


@app.command()
def list_generator_drift(
    data_dir: Annotated[Path, typer.Argument(help="The data directory of calculate", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: LogLevel = LogLevel.INFO,
    num_samples: int = 5,
    seed: int = 3003,
    expected_window=10,
    ## options are sample or pool
    type: Annotated[str, typer.Option(help="Type of list to generate", show_default=True)] = "sample",
):
    ## Input: data_dir, is output of calculate. It will have a /column directory. Each directory in /column/XXX has a bunch of .csv files.
    ## Output: output folder which will have a folder for each XXX in /column. Each folder will have a list of .csv files

    args = locals()

    global_start_time = default_timer()

    general_set_seed(seed)

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    column_dir = data_dir / "column"
    list_of_metrics = list(column_dir.iterdir())

    for metric in list_of_metrics:
        ## get list of csvs in that directory, append to 1 df with a new column for the name of the csv
        log.info("Metric to generate: %s", metric, tab=0)
        df = pd.DataFrame()
        for csv_path in metric.iterdir():
            ## if csv_path is a dir, continue
            if csv_path.is_dir():
                continue
            log.info("CSV: %s", csv_path, tab=1)
            temp_df = pd.read_csv(csv_path)
            temp_df["multiplier"] = csv_path.stem
            # keep columns multiplier, name
            temp_df = temp_df[["multiplier", "name"]]
            df = df._append(temp_df)

        output_dir = output / metric.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # filter for multipliers that have > 30 data only
        df_filtered = df.groupby("multiplier").filter(lambda x: len(x) > int(expected_window))
        df_filtered.reset_index(drop=True, inplace=True)

        multipliers = sorted(df_filtered["multiplier"].unique())

        ## SUDDEN DRIFT
        file_name = f"sudden_drift_{metric.stem}.nim"
        global_idx = 0
        with open(output_dir / file_name, "w") as f:
            f.write(f"! {file_name}\n")

            per_mult = int(expected_window)
            for mult in multipliers:
                df_mult = df_filtered[df_filtered["multiplier"] == mult]
                if len(df_mult) < per_mult:
                    selected = df_mult
                else:
                    selected = df_mult.sample(n=per_mult, replace=False, random_state=seed)

                for index, row in selected.iterrows():
                    f.write(f"{global_idx+1}-{mult}:   {row['name']}\n")
                    global_idx += 1

        ## Gradual drift
        file_name = f"gradual_drift_{metric.stem}.nim"
        global_idx = 0

        if len(multipliers) >= 3:
            with open(output_dir / file_name, "w") as f:
                f.write(f"! {file_name}\n")
                per_mult = int(int(expected_window) / 2)
                transition_period = 4
                ## Get 2 multipliers at once per loop
                for i in range(1, len(multipliers)):
                    mult1 = multipliers[i - 1]
                    mult2 = multipliers[i]
                    df_mult1 = df_filtered[df_filtered["multiplier"] == mult1]
                    df_mult2 = df_filtered[df_filtered["multiplier"] == mult2]

                    if len(df_mult1) < per_mult:
                        selected1 = df_mult1
                    else:
                        selected1 = df_mult1.sample(n=per_mult, replace=False, random_state=seed)

                    if len(df_mult2) < per_mult:
                        selected2 = df_mult2
                    else:
                        selected2 = df_mult2.sample(n=per_mult, replace=False, random_state=seed)

                    # write selected 1
                    for index, row in selected1.iterrows():
                        f.write(f"{global_idx+1}-{mult1}:   {row['name']}\n")
                        global_idx += 1

                    for i in range(1, transition_period):
                        trans1 = df_mult1.sample(n=transition_period - i, replace=False, random_state=seed)
                        trans2 = df_mult2.sample(n=i, replace=False, random_state=seed)
                        for index, row in trans2.iterrows():
                            f.write(f"{global_idx+1}-{mult2}:   {row['name']}\n")
                            global_idx += 1
                        for index, row in trans1.iterrows():
                            f.write(f"{global_idx+1}-{mult1}:   {row['name']}\n")
                            global_idx += 1

                for index, row in selected2.iterrows():
                    f.write(f"{global_idx+1}-{mult2}:   {row['name']}\n")
                    global_idx += 1

        ## Incremental drift
        file_name = f"incremental_drift_{metric.stem}.nim"
        global_idx = 0
        if len(multipliers) > 2:
            with open(output_dir / file_name, "w") as f:
                f.write(f"! {file_name}\n")
                per_mult = int(int(expected_window) / 2)
                per_trans = 5
                ## Get highest multplier and lowest multiplier df
                lowest_df = df_filtered[df_filtered["multiplier"] == multipliers[0]].sample(n=per_mult, replace=False, random_state=seed)
                highest_df = df_filtered[df_filtered["multiplier"] == multipliers[-1]].sample(n=per_mult, replace=False, random_state=seed)

                for index, row in lowest_df.iterrows():
                    f.write(f"{global_idx+1}-{multipliers[0]}:   {row['name']}\n")
                    global_idx += 1

                # for every mult in between, add transition period
                for i in range(1, len(multipliers) - 1):
                    mult = multipliers[i]
                    df_mult = df_filtered[df_filtered["multiplier"] == mult]
                    if len(df_mult) < per_mult:
                        selected = df_mult
                    else:
                        selected = df_mult.sample(n=per_trans, replace=False, random_state=seed)

                    for index, row in selected.iterrows():
                        f.write(f"{global_idx+1}-{mult}:   {row['name']}\n")
                        global_idx += 1

                for index, row in highest_df.iterrows():
                    f.write(f"{global_idx+1}-{multipliers[-1]}:   {row['name']}\n")
                    global_idx += 1

        ## recurring drift
        file_name = f"recurring_drift_{metric.stem}.nim"
        global_idx = 0
        if len(multipliers) >= 2:
            with open(output_dir / file_name, "w") as f:
                f.write(f"! {file_name}\n")
                per_mult = int(expected_window)
                # sample for highest and lowest
                df_lowest = df_filtered[df_filtered["multiplier"] == multipliers[0]]
                df_highest = df_filtered[df_filtered["multiplier"] == multipliers[-1]]

                if len(df_lowest) < per_mult:
                    selected_lowest = df_lowest
                else:
                    selected_lowest = df_lowest.sample(n=per_mult, replace=False, random_state=seed)

                if len(df_highest) < per_mult:
                    selected_highest = df_highest
                else:
                    selected_highest = df_highest.sample(n=per_mult, replace=False, random_state=seed)

                for index, row in selected_lowest.head(per_mult // 2).iterrows():
                    f.write(f"{global_idx+1}-{multipliers[0]}:   {row['name']}\n")
                    global_idx += 1

                for index, row in selected_highest.head(per_mult // 2).iterrows():
                    f.write(f"{global_idx+1}-{multipliers[-1]}:   {row['name']}\n")
                    global_idx += 1

                for index, row in selected_lowest.tail(per_mult // 2).iterrows():
                    f.write(f"{global_idx+1}-{multipliers[0]}:   {row['name']}\n")
                    global_idx += 1

                for index, row in selected_highest.tail(per_mult // 2).iterrows():
                    f.write(f"{global_idx+1}-{multipliers[-1]}:   {row['name']}\n")
                    global_idx += 1

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
