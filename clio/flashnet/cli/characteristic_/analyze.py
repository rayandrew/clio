import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")

import pandas as pd

import typer
import numpy as np

from clio.utils.characteristic import Characteristic, CharacteristicDict, Statistic
from clio.utils.general import parse_time
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import default_timer
from clio.utils.trace_pd import TraceWindowGeneratorContext, trace_get_labeled_paths, trace_time_window_generator

app = typer.Typer(name="Trace Characteristics -- Analyze", pretty_exceptions_enable=False)


@app.command()
def analyze(
    data_dir: Annotated[list[Path], typer.Argument(help="The data directories to use", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "10",
    profile_name: Annotated[str, typer.Option(help="The profile name to use", show_default=True)] = "profile_v1",
    # feat_name: Annotated[str, typer.Option(help="The feature name to use for prediction", show_default=True)] = "feat_v6_ts",
    # window_agg_size: Annotated[int, typer.Option(help="The window aggregation size to use for prediction (in number of I/Os)", show_default=True)] = 10,
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    window_size = parse_time(window_size)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)



    ctx = TraceWindowGeneratorContext()
    prev_data: pd.DataFrame = pd.DataFrame()
    characteristics: CharacteristicDict = CharacteristicDict()
    trace_counter = 0
    for i, (trace_name, trace_paths_list) in enumerate(trace_paths.items()):
        is_last_trace = i == len(trace_paths) - 1
        log.info("Trace name: %s, is last trace: %s", trace_name, is_last_trace, tab=1)
        initial_trace_path = trace_paths_list[0]

        next_initial_df = pd.read_csv(initial_trace_path)
        next_initial_df["original_ts_record"] = next_initial_df["ts_record"]
        if prev_data.empty:
            next_initial_df["ts_record"] += ts_offset
        else:
            log.info("Concatenating previous data with length %s", len(prev_data), tab=2)
            # get the last ts_record from prev_data
            ts_offset = prev_data["ts_record"].iloc[-1]
            next_initial_df["ts_record"] += ts_offset

        initial_df = pd.concat([prev_data, next_initial_df], ignore_index=True)
        reference = pd.DataFrame()
        window = pd.DataFrame()

        for i, ctx, curr_path, reference, window, is_interval_valid, is_last in trace_time_window_generator(
            ctx=ctx,
            window_size=window_size * 60,
            trace_paths=trace_paths_list,
            n_data=len(trace_paths_list),
            current_trace=initial_df,
            reference=reference,
            return_last_remaining_data=not is_last_trace,
            curr_count=0,
            curr_ts_record=0,
            end_ts=-1,
        ):
            if not is_interval_valid:
                continue

            trace_counter += 1
            log.info("Processing window %d", trace_counter, tab=1)
            log.info("Window columns: %s", list(window.columns), tab=2)
            n_data = len(window)
            read_count = int((window["io_type"] == 1).sum())
            write_count = n_data - read_count
            min_ts_record = int(window["ts_record"].min())
            max_ts_record = int(window["ts_record"].max())
            duration = max_ts_record - min_ts_record
            readonly_data = window[window["io_type"] == 1]
            writeonly_data = window[window["io_type"] == 0]
            log.debug("Generating size...")
            size = Statistic.generate(window["size"].values)
            log.debug("Generating read size...")
            read_size = Statistic.generate(readonly_data["size"].values)
            log.debug("Generating write size...")
            write_size = Statistic.generate(writeonly_data["size"].values)
            log.debug("Generating offset...")
            offset = Statistic.generate(window["offset"].values)
            log.debug("Generating iat...")
            iat = window["ts_record"].diff().dropna()
            iat[iat < 0] = 0
            iat = Statistic.generate(iat.values)
            read_iat = readonly_data["ts_record"].diff().dropna()
            read_iat[read_iat < 0] = 0
            read_iat = Statistic.generate(read_iat.values)
            write_iat = writeonly_data["ts_record"].diff().dropna()
            write_iat[write_iat < 0] = 0
            write_iat = Statistic.generate(write_iat.values)
            log.debug("Generating throughput...")
            throughput = Statistic.generate((window["size"] / window["latency"]).values)
            read_throughput = Statistic.generate((readonly_data["size"] / readonly_data["latency"]).values)
            write_throughput = Statistic.generate((writeonly_data["size"] / writeonly_data["latency"]).values)
            log.debug("Generating latency...")
            latency = Statistic.generate(window["latency"].values)
            read_latency = Statistic.generate(readonly_data["latency"].values)
            write_latency = Statistic.generate(writeonly_data["latency"].values)
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
            name = f"{trace_name}.idx_{i}"
            characteristics[name] = characteristic
            characteristics.to_msgpack(output / "characteristics.msgpack")
            characteristics.to_dataframe().to_csv(output / "characteristics.csv", index=False)

            # window_agg = window.copy()
            # window_agg["norm_ts_record"] = window_agg["ts_record"] - window_agg["ts_record"].min()
            # window_agg = calculate_agg(window_agg, group_col="norm_ts_record", window_size=window_agg_size)
            # log.info("Window agg columns: %s", list(window_agg.columns), tab=1)

            if is_last and not reference:
                prev_data = window.copy()
                if prev_data.empty:
                    ts_offset = max_ts_record
                log.info("End of current trace, saving remaining data with length %s", len(prev_data), tab=2)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)
        
        
    

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def spider(df, *, id_column, title=None, max_values=None, padding=1.25):
    categories = df._get_numeric_data().columns.tolist()
    data = df[categories].to_dict(orient='list')
    ids = df[id_column].tolist()
    if max_values is None:
        max_values = {key: padding*max(value) for key, value in data.items()}
        
    normalized_data = {key: np.array(value) / max_values[key] for key, value in data.items()}
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        ax.plot(angles, values, label=model_name)
        ax.fill(angles, values, alpha=0.15)
        for _x, _y, t in zip(angles, values, actual_values):
            t = f'{t:.2f}' if isinstance(t, float) else str(t)
            # print("coord", _x, _y)
            # ax.text(_x, _y, t, size='xx-small')
        # ax.ylabel
            
    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    if title is not None: plt.suptitle(title)
    
    return fig, ax

#   python -m clio.flashnet.cli.characteristic generate_plot_range \
#    "$range_list" "$char" "$output" --metric "$metric"
def generate_plot_range(
    range_list_csv: Annotated[Path, typer.Argument(help="The range list csv file to use", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    char_csv: Annotated[Path, typer.Argument(help="The characteristic csv file to use", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Argument(help="The output path to write the results to", exists=False, file_okay=False, dir_okay=True, resolve_path=True)],
    metric: Annotated[str, typer.Option(help="The metric to use", show_default=True)] = "iops",
):
    range_list = pd.read_csv(range_list_csv)

    char = pd.read_csv(char_csv)
    from matplotlib import pyplot as plt
 
    new_df = pd.DataFrame()
    for i, row in range_list.iterrows():
        if row['should_use'] == 'y':
            print("Plotting", row['start'], row['end'], row['type'])
            
            start = row['start']
            end = row['end']
            # get by indices of char (row number)
            char_range = char.loc[start:end]
            # plot the line metric
            char_range[metric].plot()
            # save the plot
            # fig wider
            plt.gcf().set_size_inches(15, 8)
            plt.title(f"{start}_{end}_{metric}", fontsize=20)
            # title bigger
            # axes bigger
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            
            # color of line bold
            plt.gca().lines[0].set_linewidth(2)
            plt.gca().lines[0].set_color('b')
            path = output / row["type"]
            path.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(path /f"{start}_{end}_{metric}.png")
            plt.close()


def generate_plot_spider(
    folder_csv_glob: Annotated[Path, typer.Argument(help="The folder csv glob to use", exists=True, file_okay=True, dir_okay=True, resolve_path=True)],
    output_folder: Annotated[Path, typer.Argument(help="The output folder to write the results to", exists=False, file_okay=False, dir_okay=True, resolve_path=True)],
):
    # List all .csv files in the folder, recursive
    csv_paths = list(folder_csv_glob.glob("**/characteristics.csv"))
    
    global_df = pd.DataFrame()
    # Make spider plot
    import re
    
    columns_avg = ['iops','iat_avg', 'reject_count', 'reject_ratio','accept_count', 'accept_ratio','read_count', 'write_count','size_avg', 'offset_avg', 'latency_avg', 'throughput_avg', 'duration']
    columns_p90 = ['iops','iat_p90', 'reject_count', 'reject_ratio','accept_count', 'accept_ratio','read_count', 'write_count','size_p90', 'offset_p90','latency_p90', 'throughput_p90', 'duration']
    columns = list(set(columns_avg + columns_p90))

    for csv_path in csv_paths:
        print("Processing", csv_path)
        # regex for start_idx _ end_idx \d_\d
        results = re.findall(r"\d+_\d+", str(csv_path))
        start_idx = results[0].split("_")[0]
        end_idx = results[0].split("_")[1]
        # find between recurring, sudden, gradual, and incremental
        drift_type = re.findall(r"recurring|sudden|gradual|incremental", str(csv_path))
        df = pd.read_csv(csv_path)
        df = df[columns]
        df['type'] = drift_type
        df["id"] = df['type'] + "_" + start_idx + "-" + end_idx
        global_df = pd.concat([global_df, df], ignore_index=True)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    global_df.to_csv(output_folder / "global_df_.csv", index=False)
    # Save the plot
    # loop for each type
    for t in global_df['type'].unique():
        df = global_df[global_df['type'] == t]
        print(df)
        fig, ax = spider(df[columns_avg + ['id']], id_column='id', title=t.capitalize() + " Average Spider Plot")
        output_path = output_folder / f"{t}_avg_spider_plot.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Spider plot saved to {output_path}")
        
        fig, ax = spider(df[columns_p90 + ['id']], id_column='id', title=t.capitalize() + " P90 Spider Plot")
        output_path = output_folder / f"{t}_p90_spider_plot.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Spider plot saved to {output_path}")
    
    
if __name__ == "__main__":
    app()
