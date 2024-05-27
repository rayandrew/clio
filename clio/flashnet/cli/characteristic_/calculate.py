import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

import typer

from clio.flashnet.cli.characteristic_.utils import mult_normalize

from clio.utils.characteristic import CharacteristicDict
from clio.utils.general import general_set_seed
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import default_timer

app = typer.Typer(name="Trace Characteristics -- Calculate", pretty_exceptions_enable=False)


@app.command()
def calculate(
    # data_dir: Annotated[list[Path], typer.Argument(help="The data directories", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    data_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    characteristic_path: Annotated[
        Path, typer.Option("--characteristic", help="The characteristic file", exists=True, file_okay=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    # feat_name: Annotated[str, typer.Option(help="The feature name", show_default=True)] = "feat_v6_ts",
    # window_agg_size: Annotated[int, typer.Option(help="The window aggregation size (in number of I/Os)", show_default=True)] = 10,
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    seed: Annotated[int, typer.Option(help="The seed to use", show_default=True)] = 3003,
):
    args = locals()

    global_start_time = default_timer()

    general_set_seed(seed)

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    characteristics = CharacteristicDict.from_msgpack(characteristic_path)
    log.info("Loaded characteristics from %s", characteristic_path, tab=0)

    # for name, value in characteristics.items():
    #     log.info("Name: %s", name, tab=1)

    characteristics_df = characteristics.to_dataframe()
    log.info("Characteristics dataframe shape: %s", characteristics_df.shape, tab=0)
    # log.info("Characteristics dataframe columns: %s", list(characteristics_df.columns), tab=0)
    # find pairwise characteristics multiplication

    base_column_dir = output / "column"

    base_column_dir.mkdir(parents=True, exist_ok=True)

    data_dict: dict[int, dict[int, pd.DataFrame]] = {}
    names: set[str] = set()

    CHARACTERISTIC_COLUMNS = [
        # read
        # "read_size_avg",
        # "read_latency_avg",
        # "read_iat_avg",
        # "read_throughput_avg",
        # "read_size_median",
        # "read_latency_median",
        # "read_iat_median",
        # "read_throughput_median",
        # general
        # "iat_avg",
        # "latency_avg",
        "num_io",
        # "size_avg",
        # "throughput_avg",
        # "iat_median",
        # "latency_median",
        # "size_median",
        # "throughput_median",
        # write
        # "write_size_avg",
        # "write_latency_avg",
        # "write_iat_avg",
        # "write_throughput_avg",
        # "write_count",
        # "rw_ratio",
        # "write_size_median",
        # "write_latency_median",
        # "write_iat_median",
        # "write_throughput_median",
    ]

    # # filter for write_ratio > 0.6
    # log.info(f"Filtering for write_ratio < 0.6, length before: {len(characteristics_df)}", tab=0)
    # # print write_ratio statistics like min, max, etc
    # log.info("Write ratio statistics: %s", characteristics_df["write_ratio"].describe(), tab=1)
    # characteristics_df = characteristics_df[characteristics_df["write_ratio"] < 0.6]
    # log.info(f"Length after: {len(characteristics_df)}", tab=0)
    characteristics_df = characteristics_df[characteristics_df["read_count"] > 100000]
    characteristics_df = mult_normalize(characteristics_df)

    # pairwise window that has size vs 2*size vs 3*size and so on
    for column in CHARACTERISTIC_COLUMNS:
        char_df = characteristics_df.copy()
        base_df = char_df[char_df[column] == 1]

        mult_dict: dict[int, pd.DataFrame] = {
            1: base_df,
        }
        for mult in [1.2, 1.5, 2, 2.4, 2.8, 3.5, 4, 5, 6, 7, 8, 9, 10]:
            # find the window that has roughly equal to size * mult
            tol = 0.1
            mult_df = char_df[(char_df[column] > mult) & (char_df[column] <= mult + tol)]
            if mult_df.empty:
                log.info("No window found for %s_mult_%s", column, mult, tab=1)
                continue
            mult_dict[mult] = mult_df
            names.update(mult_df["name"])

        # check if the mult_dict contains only the base
        if len(mult_dict) == 1:
            continue

        # log.info("Names: %s", names, tab=0)

        column_dir = base_column_dir / column

        # NOTE: REMOVE THIS
        # if column_dir.exists():
        #     log.warning("Column directory %s already exists, skipping it...", column_dir, tab=0)
        #     continue

        column_dir.mkdir(parents=True, exist_ok=True)

        log.info("Column: %s", column, tab=1)
        for k, v in mult_dict.items():
            log.info("Key: %s, shape: %s", k, v.shape, tab=2)
            v.to_csv(column_dir / f"{k}.csv", index=False)

        data_dict[column] = mult_dict

    names = sorted(names)

    N_COL = 2
    # create a heatmap for each column
    # for column, v in data_dict.items():
    #     column_dir = base_column_dir / column
    #     log.info("Column: %s", column, tab=0)
    #     len_data = len(v)
    #     n_col = N_COL
    #     n_row = (len_data // n_col) + 1
    #     # if n_row == 1:
    #     #     n_row = 2
    #     log.info("n_row: %s, n_col: %s", n_row, n_col, tab=1)
    #     fig = plt.figure(figsize=(n_col * 6, n_row * 6))
    #     gs = GridSpec(n_row, n_col, figure=fig)
    #     fig.suptitle(f"Column: {column}")
    #     for i, (mult, v2) in enumerate(v.items()):
    #         # fig, ax = plt.subplots(figsize=(6, 6))
    #         # ax = axs[i // n_col, i % n_col]
    #         ax = fig.add_subplot(gs[i // n_col, i % n_col])
    #         v2_cleaned = v2.drop(columns=["name", "disks", "ts_unit", "size_unit"])  # , "start_ts", "end_ts","duration"])
    #         # pick only _avg columns and num_io, iops
    #         cols = [col for col in v2_cleaned.columns if "_avg" in col]
    #         cols.append("num_io")
    #         cols.append("iops")
    #         v2_cleaned = v2_cleaned[cols]
    #         # v2_cleaned = v2_cleaned.filter(like="_avg")
    #         # v2_cleaned = v2_cleaned.drop(columns=v2_cleaned.columns)
    #         corr = v2_cleaned.corr()
    #         log.info("Corr columns: %s", list(corr.columns), tab=1)
    #         sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    #         ax.set_title(f"Mult: {mult}")
    #         # fig.suptitle(f"Column: {column}")
    #         # fig.tight_layout()
    #         # fig.savefig(column_dir / f"heatmap_{i}.png")
    #         # plt.close(fig)
    #     fig.tight_layout()
    #     fig.savefig(column_dir / "heatmap.png")
    #     plt.close(fig)

    # for column, v in data_dict.items():
    #     for mult, v2 in v.items():
    #         log.info("Column: %s, Mult: %s, Shape: %s", column, mult, v2.shape, tab=1)

    # pairwise plot between the base and the mult
    return
    sns.set_theme(font_scale=1.5)
    for column, v in data_dict.items():
        column_dir = base_column_dir / column
        column_plot_dir = column_dir / "plot"
        # column_plot_dir.mkdir(parents=True, exist_ok=True)
        log.info("Column: %s", column, tab=0)
        for mult, v2 in v.items():
            if mult == 1:
                continue

            mult_plot_dir = column_plot_dir / f"mult_{mult}"

            if mult_plot_dir.exists():
                log.warning("Mult plot directory %s already exists, skipping it...", mult_plot_dir, tab=1)
                continue

            mult_plot_dir.mkdir(parents=True, exist_ok=True)
            log.info("Mult: %s", mult, tab=1)
            base_df = v[1]
            mult_df = v[mult]
            #
            # cols.append("num_io")
            # cols.append("iops")
            # base_df = base_df[cols]
            # base_df["mult"] = 1
            # mult_df = mult_df[cols]
            # mult_df["mult"] = mult

            # base df will only have 1 row
            assert len(base_df) == 1

            base_df_name = base_df["name"].values[0]
            base_df_data = pd.read_csv(data_dir / f"{base_df_name}.csv")
            base_df_data["iat"] = base_df_data["ts_record"].diff().dropna()
            base_df_data.loc[base_df_data["iat"] < 0, "iat"] = 0
            base_df_data["size"] = base_df_data["size"] / 1_000_000  # convert to MB
            base_df_data["latency"] = base_df_data["latency"] / 1000  # convert to ms
            base_df_data["throughput"] = base_df_data["size"] / base_df_data["latency"] * 1000  # MB/s
            base_df_data = base_df_data[base_df_data["latency"] <= 2.3]  # remove outliers
            base_df_data["mult"] = 1
            base_df_data = base_df_data.drop(
                columns=[
                    "ts_record",
                    "ts_submit",
                    "original_ts_record",
                    "size_after_replay",
                ]
            )
            base_df_data = base_df_data.reset_index(drop=True)
            # base_df_data = base_df_data.reset_index(drop=True)

            base_df_data_writeonly = base_df_data[base_df_data["io_type"] == 0].drop(columns=["io_type"])
            if len(base_df_data_writeonly) > 50000:
                base_df_data_writeonly = base_df_data_writeonly.sample(50000)
            base_df_data_readonly = base_df_data[base_df_data["io_type"] == 1].drop(columns=["io_type"])
            if len(base_df_data_readonly) > 50000:
                base_df_data_readonly = base_df_data_readonly.sample(50000)

            for name in list(mult_df["name"].unique()):
                m_df = mult_df[mult_df["name"] == name]
                assert len(m_df) == 1

                m_df_name = m_df["name"].values[0]
                m_df_data = pd.read_csv(data_dir / f"{m_df_name}.csv")
                m_df_data["iat"] = m_df_data["ts_record"].diff().dropna()
                m_df_data.loc[m_df_data["iat"] < 0, "iat"] = 0
                m_df_data["size"] = m_df_data["size"] / 1_000_000  # convert to MB
                m_df_data["latency"] = m_df_data["latency"] / 1000  # convert to ms
                m_df_data = m_df_data[m_df_data["latency"] <= 2.3]  # remove outliers
                m_df_data["throughput"] = m_df_data["size"] / m_df_data["latency"] * 1000  # MB/s
                m_df_data["mult"] = mult
                m_df_data = m_df_data.drop(
                    columns=[
                        "ts_record",
                        "ts_submit",
                        "original_ts_record",
                        "size_after_replay",
                    ]
                )
                m_df_data = m_df_data.reset_index(drop=True)
                # m_df_data = m_df_data.reset_index(drop=True)
                m_df_data_writeonly = m_df_data[m_df_data["io_type"] == 0].drop(columns=["io_type"])
                if len(m_df_data_writeonly) > 50000:
                    m_df_data_writeonly = m_df_data_writeonly.sample(50000)
                m_df_data_readonly = m_df_data[m_df_data["io_type"] == 1].drop(columns=["io_type"])
                if len(m_df_data_readonly) > 50000:
                    m_df_data_readonly = m_df_data_readonly.sample(50000)

                # base_df_data_ = base_df_data.drop(columns=["ts_record", "original_ts_record"])
                # m_df_data_ = m_df_data.drop(columns=["ts_record", "original_ts_record"])

                # g = sns.pairplot(
                #     pd.concat([base_df_data.sample(5000), m_df_data.sample(5000)]).reset_index(drop=True),
                #     diag_kind="kde",
                #     hue="mult",
                #     palette="tab10",
                # )
                # sns.move_legend(g, "upper right", bbox_to_anchor=(1.0, 1.0))
                # g.figure.suptitle(f"Base (1x) vs Mult ({mult}x), Column: {column}", y=1.05)
                # g.figure.savefig(mult_plot_dir / f"pairplot_{name}.png", dpi=300)
                # plt.close(g.figure)

                fig = plt.figure(figsize=(12, 12))

                fig.suptitle(
                    "\n".join([f"Base (1x) vs Mult ({mult}x)", f"Column: {column}", f"Base name: {base_df_name}", f"Mult name: {m_df_name}"]),
                    fontsize=14,
                )
                n_col = 3
                n_row = 12 // n_col
                gs = GridSpec(n_row, n_col, figure=fig)

                # plotting latency
                ax = fig.add_subplot(gs[0, 0])
                sns.kdeplot(base_df_data["latency"], ax=ax, label="Base")
                sns.kdeplot(m_df_data["latency"], ax=ax, label="Mult")
                ax.set_title("Latency")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Density")
                # ax.set_xlim(0, 1.0)

                # plotting latency read
                ax = fig.add_subplot(gs[0, 1])
                sns.kdeplot(base_df_data_readonly["latency"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_readonly["latency"], ax=ax, label="Mult")
                ax.set_title("Latency Read")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Density")
                # ax.set_xlim(0, 1.0)

                ax = fig.add_subplot(gs[0, 2])
                sns.kdeplot(base_df_data_writeonly["latency"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_writeonly["latency"], ax=ax, label="Mult")
                ax.set_title("Latency Write")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Density")
                # ax.set_xlim(0, 1.0)

                ax = fig.add_subplot(gs[1, 0])
                sns.kdeplot(base_df_data["size"], ax=ax, label="Base")
                sns.kdeplot(m_df_data["size"], ax=ax, label="Mult")
                ax.set_title("Size")
                ax.set_xlabel("Size (MB)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[1, 1])
                sns.kdeplot(base_df_data_readonly["size"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_readonly["size"], ax=ax, label="Mult")
                ax.set_title("Size Read")
                ax.set_xlabel("Size (MB)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[1, 2])
                sns.kdeplot(base_df_data_writeonly["size"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_writeonly["size"], ax=ax, label="Mult")
                ax.set_title("Size Write")
                ax.set_xlabel("Size (MB)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[2, 0])
                sns.kdeplot(base_df_data["iat"], ax=ax, label="Base")
                sns.kdeplot(m_df_data["iat"], ax=ax, label="Mult")
                ax.set_title("IAT")
                ax.set_xlabel("IAT (ms)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[2, 1])
                sns.kdeplot(base_df_data_readonly["iat"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_readonly["iat"], ax=ax, label="Mult")
                ax.set_title("IAT Read")
                ax.set_xlabel("IAT (ms)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[2, 2])
                sns.kdeplot(base_df_data_writeonly["iat"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_writeonly["iat"], ax=ax, label="Mult")
                ax.set_title("IAT Write")
                ax.set_xlabel("IAT (ms)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[3, 0])
                sns.kdeplot(base_df_data["throughput"], ax=ax, label="Base")
                sns.kdeplot(m_df_data["throughput"], ax=ax, label="Mult")
                ax.set_title("Throughput")
                ax.set_xlabel("Throughput (MB/s)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[3, 1])
                sns.kdeplot(base_df_data_readonly["throughput"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_readonly["throughput"], ax=ax, label="Mult")
                ax.set_title("Throughput Read")
                ax.set_xlabel("Throughput (MB/s)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[3, 2])
                sns.kdeplot(base_df_data_writeonly["throughput"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_writeonly["throughput"], ax=ax, label="Mult")
                ax.set_title("Throughput Write")
                ax.set_xlabel("Throughput (MB/s)")

                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0))

                fig.tight_layout()
                fig.savefig(mult_plot_dir / f"plot.base_vs_mult_{mult}.base_{base_df_name}.mult_{m_df_name}.png", dpi=300)
                plt.close(fig)

                # gmm_base_df = GaussianMixture(n_components=4, random_state=seed)
                # values = base_df_data_readonly["latency"].values
                # gmm_base_df.fit(values.reshape(-1, 1))

                # # predict the mult dataset
                # m_df_data_readonly["cluster"] = gmm_base_df.predict(m_df_data_readonly["latency"].values.reshape(-1, 1))

                # # plot cluster
                # fig, ax = plt.subplots(figsize=(10, 5))
                # ax.set_title(f"Base (1x) vs Mult ({mult}x), Column: {column}, GMM")
                # ax.set_xlabel("Latency (ms)")
                # ax.set_ylabel("Size (MB)")
                # sns.scatterplot(
                #     x="latency",
                #     y="size",
                #     hue="cluster",
                #     data=m_df_data_readonly,
                #     palette="tab10",
                #     style="reject",
                #     ax=ax,
                # )
                # # ax.set_xlim(0, 0.5)
                # fig.tight_layout()
                # fig.savefig(mult_plot_dir / f"cluster_base.mult_{mult}.base_{base_df_name}.mult_{m_df_name}.gmm.png", dpi=300)
                # plt.close(fig)

                # gmm_m_df = GaussianMixture(n_components=4, random_state=seed)
                # values = m_df_data_readonly["latency"].values
                # gmm_m_df.fit(values.reshape(-1, 1))

                # m_df_data_readonly["cluster"] = gmm_m_df.predict(m_df_data_readonly["latency"].values.reshape(-1, 1))

                # # plot cluster
                # fig, ax = plt.subplots(figsize=(10, 5))
                # ax.set_title(f"Base (1x) vs Mult ({mult}x), Column: {column}, GMM")
                # ax.set_xlabel("Latency (ms)")
                # ax.set_ylabel("Size (MB)")
                # # markers based on reject (1 or 0)
                # sns.scatterplot(
                #     x="latency",
                #     y="size",
                #     hue="cluster",
                #     data=m_df_data_readonly,
                #     style="reject",
                #     palette="tab10",
                #     ax=ax,
                # )
                # # ax.set_xlim(0, 0.5)
                # fig.tight_layout()
                # fig.savefig(mult_plot_dir / f"cluster_mult.mult_{mult}.base_{base_df_name}.mult_{m_df_name}.gmm.png", dpi=300)
                # plt.close(fig)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title(f"Readonly Data. Base (1x) vs Mult ({mult}x), Column: {column}, Size vs Latency")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Size (MB)")
                sns.scatterplot(
                    x="latency",
                    y="size",
                    hue="reject",
                    data=m_df_data_readonly,
                    palette="tab10",
                    style="reject",
                    ax=ax,
                )
                # ax.set_xlim(0, 0.5)
                fig.tight_layout()
                fig.savefig(mult_plot_dir / f"scatter.base.mult_{mult}.base_{base_df_name}.mult_{m_df_name}.size_latency.readonly.png", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title(f"Combined. Base (1x) vs Mult ({mult}x), Column: {column}, Size vs Latency")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Size (MB)")
                sns.scatterplot(
                    x="latency",
                    y="size",
                    hue="reject",
                    data=m_df_data,
                    palette="tab10",
                    style="reject",
                    ax=ax,
                )
                # ax.set_xlim(0, 0.5)
                fig.tight_layout()
                fig.savefig(mult_plot_dir / f"scatter.base.mult_{mult}.base_{base_df_name}.mult_{m_df_name}.size_latency.png", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title(f"Readonly Data. Base (1x) vs Mult ({mult}x), Column: {column}, Size vs Throughput")
                ax.set_xlabel("Throughput (MB/s)")
                ax.set_ylabel("Size (MB)")
                sns.scatterplot(
                    x="throughput",
                    y="size",
                    hue="reject",
                    data=m_df_data_readonly,
                    palette="tab10",
                    style="reject",
                    ax=ax,
                )
                # ax.set_xlim(0, 0.5)
                fig.tight_layout()
                fig.savefig(mult_plot_dir / f"scatter.base.mult_{mult}.base_{base_df_name}.mult_{m_df_name}.size_throughput.readonly.png", dpi=300)

                break

            with open(mult_plot_dir / "mult-file.txt", "w") as f:
                f.write("! Base\n")
                f.write(base_df_name)
                f.write("\n")
                f.write("\n")
                f.write(f"! Multiplier {mult}")
                f.write("\n")
                for name in list(mult_df["name"].unique()):
                    f.write(name)
                    f.write("\n")

            # df = pd.concat([base_df, mult_df])

            # fig = plt.figure(figsize=(10, 10))
            # n_col = 4
            # n_row = (len(df.columns) // n_col) + 1
            # gs = GridSpec(n_row, n_col, figure=fig)
            # fig.suptitle(f"Column: {column}, Mult: {mult}")

            # # generate barplot of multiplier of each column
            # for i, col in enumerate(df.columns):
            #     if col == "mult":
            #         continue
            #     ax = fig.add_subplot(gs[i // n_col, i % n_col])
            #     sns.barplot(x="mult", y=col, data=df, ax=ax, hue="mult", palette="tab10")
            #     # remove legend
            #     ax.get_legend().remove()
            #     ax.set_title(col)
            #     ax.set_xlabel("Multiplier")
            #     ax.set_ylabel("")
            #     # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            #     # ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            #     ax.legend

            # fig.tight_layout()
            # fig.savefig(column_dir / f"barplot_{mult}.png")

            # g = sns.pairplot(df, diag_kind="kde", hue="mult", palette="tab10")
            # sns.move_legend(g, "upper right", bbox_to_anchor=(1.0, 1.0))
            # g.figure.suptitle(f"Column: {column}, Mult: {mult}", y=1.05)
            # g.figure.savefig(column_dir / f"pairplot_{mult}.png", dpi=300)

            # for col in base_df.columns:
            #     if col == "mult":
            #         continue
            #     log.info("Column: %s", col, tab=1)
            #     stat, p = kruskal(base_df[col], mult_df[col])
            #     log.info("Kruskal stats: %s, p-value: %s", stat, p, tab=1)
            #     log.info("Different distribution: %s", p < 0.05, tab=1)
            #     stat, p = mannwhitneyu(base_df[col], mult_df[col])
            #     log.info("Mannwhitneyu stats: %s, p-value: %s", stat, p, tab=1)
            #     log.info("Different distribution: %s", p < 0.05, tab=1)

    windows_idx: dict[str, list[int]] = {}
    for name in names:
        # <name>.idx_<idx> and <name> might contain "." in the name
        name, idx = name.split(".idx_")
        idx = int(idx)
        if name not in windows_idx:
            windows_idx[name] = []
        windows_idx[name].append(idx)

    windows_idx = {k: sorted(v) for k, v in windows_idx.items()}
    # window_dir = output / "window"
    # window_dir.mkdir(parents=True, exist_ok=True)
    for name, idxs in windows_idx.items():
        log.info("Name: %s, Idxs: %s", name, idxs, tab=1)

        # ctx = TraceWindowGeneratorContext()
        # window_size = 1
        # trace_paths_list = trace_get_labeled_paths(
        #     data_dir / name,
        #     profile_name="profile_v1",
        # )
        # initial_df = pd.read_csv(trace_paths_list[0])
        # reference = pd.DataFrame()
        # for i, ctx, curr_path, reference, window, is_interval_valid, is_last in trace_time_window_generator(
        #     ctx=ctx,
        #     window_size=window_size * 60,
        #     trace_paths=trace_paths_list,
        #     n_data=len(trace_paths_list),
        #     current_trace=initial_df,
        #     reference=reference,
        #     return_last_remaining_data=True,
        #     curr_count=0,
        #     curr_ts_record=0,
        #     end_ts=-1,
        # ):
        #     if not is_interval_valid:
        #         continue

        #     if i in idxs:
        #         log.info("Name: %s, Idx: %s, Window shape: %s", name, i, window.shape, tab=1)
        #         window.to_csv(window_dir / f"{name}.idx_{i}.csv", index=False)

        # window_agg = window.copy()
        # window_agg["norm_ts_record"] = window_agg["ts_record"] - window_agg["ts_record"].min()
        # window_agg = calculate_agg(window_agg, group

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
