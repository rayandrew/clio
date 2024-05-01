#!/usr/bin/env python3

import argparse
import bisect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc

import matplotlib.pyplot as plt

import clio.flashnet.ip_finder as ip_finder
from clio.flashnet.constants import N_FUTURE, N_HISTORY, THPT_DROP_RATE

from clio.utils.logging import log_get

log = log_get(__name__)


# Vectorized way to collect historical data
def collect_history(df: pd.DataFrame, col_name: str, n_history: int):
    df = df.copy()
    for i in range(1, n_history + 1):
        df["hist_" + str(i - 1) + "_" + col_name] = df[col_name].shift(i).fillna(0)
    return df


# Vectorized way to collect future data
def collect_future(df: pd.DataFrame, col_name: str, n_future: int):
    df = df.copy()
    for i in range(0, n_future):
        df["future_" + str(i) + "_" + col_name] = df[col_name].shift(-i).fillna(0)
    return df


# # FEATURE 1: add N last latency
# def collect_history_v1(df: pd.DataFrame, col_name: str, n_history: int = N_HISTORY):
#     history_holder = pd.DataFrame()
#     for n in range(1, n_history + 1):
#         history_holder[n] = df[col_name].shift(n).fillna(0)
#     history_holder["all_history"] = history_holder.apply(list, axis=1)
#     return history_holder["all_history"].values


# # FEATURE 1: add N future entry (The current IO is regarded as part of the future)
def collect_future_v1(df: pd.DataFrame, col_name: str, n_future: int = N_FUTURE):
    future_holder = pd.DataFrame()
    for n in range(0, n_future):
        future_holder[n] = df[col_name].shift(-n).fillna(0)
    future_holder["all_future"] = future_holder.apply(list, axis=1)
    return future_holder["all_future"].values


def mark_possible_start_vectorized(df: pd.DataFrame, ip_latency_threshold: float, ip_throughput_threshold: float, thpt_drop_rate: float):
    df = df.copy()
    # Vectorize conditions
    condition_fast = (df["throughput"] > ip_throughput_threshold) | (df["latency"] < ip_latency_threshold)
    log.info("Throughput drop %s", df["throughput_drop"].mean())
    condition_gc_start = df["throughput_drop"] >= thpt_drop_rate

    log.info("Fast IO = %s", condition_fast.sum())
    log.info("GC-Start = %s", condition_gc_start.sum())

    # Assign values based on conditions
    df["mark_start1"] = " "  # Default assignment
    df.loc[~condition_fast & condition_gc_start, "mark_start1"] = " GC-Start1 "
    # df.loc[condition_gc_start, "mark_start1"] = " GC-Start1 "
    return df


# def find_gc_end_v1(df: pd.DataFrame, median_throughput: float):
#     assert "mark_start1" in df.columns, "mark_start1 column is missing"

#     df = df.copy()

#     df["mark_gc_end"] = "  "  # Direct assignment to all rows
#     df["mark_tail"] = ""  # Initialize mark_tail column

#     # Identify rows where GC starts
#     gc_starts = df["mark_start1"] == " GC-Start1 "
#     log.info("GC starts = %s", gc_starts.sum())

#     # Iterate over each GC start point
#     for idx in df.index[gc_starts]:
#         df.at[idx, "mark_tail"] = " Tail-Period "  # Mark the GC start as Tail-Period
#         # Check for the condition in future rows
#         subsequent_indexes = range(idx + 1, len(df))
#         for j in subsequent_indexes:
#             if all(df.at[j, "n_future_throughput"] >= median_throughput):
#                 break  # Exit the inner loop if condition is met
#             df.at[j, "mark_tail"] = " Tail-Period "  # Otherwise, mark as Tail-Period

#     # The count of slow I/O operations
#     n_slow_io = (df["mark_tail"] == " Tail-Period ").sum()

#     return df, n_slow_io


def find_gc_end(df: pd.DataFrame, median_throughput: float, n_future: int = N_FUTURE):
    assert "mark_start1" in df.columns, "mark_start1 column is missing"
    for i in range(n_future):
        assert f"future_{i}_throughput" in df.columns, f"future_{i}_throughput column is missing"

    df = df.copy()

    df["mark_gc_end"] = "  "  # Direct assignment to all rows
    df["mark_tail"] = ""  # Initialize mark_tail column

    # Identify rows where GC starts
    gc_starts = df["mark_start1"] == " GC-Start1 "
    # log.info("GC starts = %s", gc_starts.sum())
    # idx_gc_start1 = df.index[gc_starts].tolist()
    # log.info("GC-Start1 head: %s", idx_gc_start1[:20])
    # log.info("GC-Start1 tail: %s", idx_gc_start1[-20:])

    # Iterate over each GC start point
    n_slow_io = 0
    max_idx = len(df) - 1
    # idxs = {}
    idxs = []
    current_idx = 0
    for idx in df.index[gc_starts]:
        # if idx in idxs and idxs[idx]:
        #     continue
        # idxs[idx] = True
        if idx < current_idx:
            # log.info("IDX %s, current_idx %s", idx, current_idx)
            continue
        # log.info("IDX: %s", idx)
        n_slow_io += 1
        df.at[idx, "mark_tail"] = " Tail-Period "  # Mark the GC start as Tail-Period
        current_idx = idx
        idxs.append(idx)
        # Check for the condition in future rows
        # subsequent_indexes = range(idx + 1, max_idx)
        # cond = False
        while idx < max_idx:
            # cond = True
            idx += 1
            # j = i
            # for j in subsequent_indexes:
            # idxs[j] = True
            # log.info("Future 0 throughput: %s", df.at[idx, "future_0_throughput"])
            #     break
            if all(df.at[idx, f"future_{i}_throughput"] >= median_throughput for i in range(n_future)):
                # if all(i >= median_throughput for i in row["n_future_throughput"]):
                # if Yes, it is the GC-END; no need to mark anything
                break
            else:
                # idxs[j] = True
                n_slow_io += 1
                # idxs.append(j)
                df.at[idx, "mark_tail"] = " Tail-Period "  # Otherwise, mark as Tail-Period

        current_idx = idx + 1

    # =========================

    # idx = 0
    # while idx <= max_idx:
    #     row = df.iloc[idx]
    #     # Will start processing at " GC-Start " marker
    #     if row["mark_start1"] == " GC-Start1 ":
    #         idxs.append(idx)
    #         n_slow_io += 1
    #         df.at[idx, "mark_tail"] = " Tail-Period "  # Mark the START
    #         # going down checking the future thpt
    #         while idx < max_idx:
    #             idx += 1
    #             row = df.iloc[idx]
    #             # idxs.append(idx)
    #             if all(df.at[idx, f"future_{i}_throughput"] >= median_throughput for i in range(n_future)):
    #                 # if Yes, it is the GC-END; no need to mark anything
    #                 break
    #             else:
    #                 n_slow_io += 1
    #                 # idxs.append(idx)
    #                 df.at[idx, "mark_tail"] = " Tail-Period "  # Mark it as slow
    #     # check next row until finding the starting point of the GC
    #     idx += 1

    # =========================

    # idxss = list(idxs.keys())
    idxss = idxs
    n = 50
    log.info("")
    log.info("idxs head: %s", idxss[:n])
    log.info("idxs middle: %s", idxs[len(idxs) // 2 - (n // 2) : len(idxs) // 2 + (n // 2)])
    log.info("idxs tail: %s", idxss[-n:])

    # sys.exit(0)
    # The count of slow I/O operations
    # n_slow_io += (df["mark_tail"] == " Tail-Period ").sum()

    return df, n_slow_io


def calc_percent(partition: float, total: float, precision: int = 2):
    return str(round(partition * 100 / total, precision)) + "%"


def filter_outlier(df: pd.DataFrame, median_latency: float, median_throughput: float, ip_latency_threshold: float):
    df = df.copy()
    # Marking outlier1
    df["mark_outlier"] = df.apply(lambda row: "  ", axis=1)

    condition1 = (df["mark_tail"] == " Tail-Period ") & (df["latency"] <= median_latency) & (df["throughput"] >= median_throughput)
    df.loc[condition1, "mark_outlier"] = " outlier1 "

    # Marking outlier2
    condition2 = (df["mark_tail"] != " Tail-Period ") & (df["latency"] > ip_latency_threshold) & (df["throughput"] < median_throughput)
    df.loc[condition2, "mark_outlier"] = " outlier2 "

    # Counting outliers
    n_outlier1 = condition1.sum()
    n_outlier2 = condition2.sum()

    # Remove outliers marked as 'outlier1' and 'outlier2'
    df = df[df["mark_outlier"] == "  "]
    df = df.reset_index(drop=True)

    # Step 1: Identify rows with ' Tail-Period '
    df["is_tail_period"] = df["mark_tail"] == " Tail-Period "

    # Step 2: Create a group identifier for contiguous ' Tail-Period ' blocks
    df["tail_group"] = (df["is_tail_period"] != df["is_tail_period"].shift()).cumsum()

    # Only consider groups that are ' Tail-Period ' and calculate their sizes
    tail_group_sizes = df[df["is_tail_period"]].groupby("tail_group").size()

    # Filter groups where size <= N_HISTORY
    small_tail_groups = tail_group_sizes[tail_group_sizes <= N_HISTORY].index

    # Step 3: Mark rows in those groups as outliers
    df.loc[df["tail_group"].isin(small_tail_groups) & df["is_tail_period"], "mark_outlier"] = " outlier3 "
    n_outlier3 = df["mark_outlier"].eq(" outlier3 ").sum()

    df = df[df["mark_outlier"] == "  "]
    df = df.reset_index(drop=True)

    df = df.drop(["mark_tail", "is_tail_period", "tail_group", "mark_outlier"], axis=1, errors="ignore")

    return df, n_outlier1, n_outlier2, n_outlier3


def calc_cdf_gain(y_raw: np.ndarray, y_best: np.ndarray) -> str:
    # 1. The Raw Value
    # sort the data in ascending order
    N = len(y_raw)
    x_2 = np.sort(y_raw)
    y_2 = np.arange(N) / float(N)

    # Must limit the x axis, we don't want to calculate the area of the insignificant tail
    p70_lat = np.percentile(y_raw, 70)
    x_limiter = max(p70_lat * 3, 1000)  # same as how we draw the CDF above
    idx = bisect.bisect_left(x_2, x_limiter)
    x_2 = x_2[:idx]
    y_2 = y_2[:idx]
    max_tail_value = x_2[-1]

    # 2. The BEST-case Value
    # sort the data in ascending order
    N = len(y_best)
    x_1 = np.sort(y_best)
    y_1 = np.arange(N) / float(N)

    # Must limit the x axis
    idx = bisect.bisect_left(x_1, x_limiter)
    x_1 = x_1[:idx]
    y_1 = y_1[:idx]

    # Must add padding to make sure that it reach the x limit
    if max(x_1) < max_tail_value:
        x_1 = np.append(x_1, max_tail_value)
        y_1 = np.append(y_1, 1)

    # 3. Calculate the AUC
    cdf_raw = auc(x_2, y_2)
    cdf_best = auc(x_1, y_1)

    percent_gain = calc_percent(cdf_best - cdf_raw, cdf_raw)

    return percent_gain


def labeling(data: pd.DataFrame, filter_outlier: bool = False) -> pd.DataFrame:
    df = data.copy()
    if "size_after_replay" not in df.columns:
        df["size_after_replay"] = df["size"]
    df = df[["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "size_after_replay"]]

    # 1. Add more variable to Analyze the Trace
    stats_total_io = len(df)
    stats_n_read = len(df[df["io_type"] == 1])

    # Sort based on ts_submit, there is a slight out of order due to multithreading submission
    df = df.sort_values("ts_submit")
    df = df.reset_index(drop=True)

    # put a separator
    df["sep"] = "  "

    # add throughput
    df["throughput"] = df["size"] / df["latency"]
    df["throughput"] = df["throughput"].round(0)

    # put a separator
    df["sep2"] = "  "

    # 2. Find SLOW throughput and Get the Latency Threshold
    ip_latency_threshold, ip_latency_percent = ip_finder.tangent_based(df["latency"])
    # if throughput is higher, it is definitely FAST IO
    ip_throughput_threshold, ip_thpt_percent = ip_finder.tangent_based(df["throughput"])

    if ip_latency_percent < 50:
        log.warning("WARNING: this trace profile is BAD because the IP latency is < 50%. Flashnet won't be able to make any significant improvement.")

    # slow throughput if it is less than the median throughput
    median_throughput = np.percentile(df["throughput"], 50)
    # if less than median_latency, the IO is FAST
    median_latency = np.percentile(df["latency"], 50)
    log.info("IP latency threshold : " + str(ip_latency_threshold) + " (" + str(ip_latency_percent) + "%)")
    log.info("Median throughput threshold : " + str(median_throughput) + " (" + str(50) + "%)")
    log.info("IP throughput threshold : " + str(ip_throughput_threshold) + " (" + str(ip_thpt_percent) + "%)")
    log.info("Median latency threshold : " + str(median_latency) + " (" + str(50) + "%)")

    # 3. Find the GC-Start
    df = collect_history(df, "throughput", n_history=N_HISTORY)
    df = collect_history(df, "latency", n_history=N_HISTORY)
    df = collect_future(df, "throughput", n_future=N_FUTURE)
    df["throughput_drop"] = round(df["hist_0_throughput"] / (df["throughput"] + 0.1), 1)
    # df["n_hist_throughput"] = collect_history_v1(df, "throughput", n_history=N_HISTORY)
    # df["n_hist_latency"] = collect_history_v1(df, "latency", n_history=N_HISTORY)
    # df["n_future_throughput"] = collect_future_v1(df, "throughput", n_future=N_FUTURE)
    # hist_df = pd.DataFrame(df["n_hist_throughput"].tolist(), index=df.index)
    # first_hist_throughput = hist_df[0].fillna(0)
    # df["throughput_drop"] = round(first_hist_throughput / (df["throughput"] + 0.1), 1)

    # dfff = df.copy()
    # dfff["n_hist_throughput"] = collect_history_v1(dfff, "throughput", n_history=N_HISTORY)
    # log.info("n_hist_throughput = %s", dfff["n_hist_throughput"][0])
    # log.info("hist_0_throughput = %s", dfff["hist_0_throughput"].values)
    # hist_df = pd.DataFrame(dfff["n_hist_throughput"].tolist(), index=df.index)
    # first_hist_throughput = hist_df[0].fillna(0)
    # log.info("First hist throughput = %s", first_hist_throughput.values)
    # dfff["throughput_drop"] = round(first_hist_throughput / (df["throughput"] + 0.1), 1)

    # log.info("1 -- Throughput drop %s", df["throughput_drop"].mean())
    # log.info("2 -- Throughput drop %s", dfff["throughput_drop"].mean())

    df = mark_possible_start_vectorized(
        df, ip_latency_threshold=ip_latency_threshold, ip_throughput_threshold=ip_throughput_threshold, thpt_drop_rate=THPT_DROP_RATE
    )
    log.info("GC start %s", df["mark_start1"].value_counts())
    # df["n_future_throughput"] = collect_future_v1(df, "throughput", n_future=N_FUTURE)

    # compare the two methods
    # log.info("n_future_throughput = %s", df["n_future_throughput"].values[0])
    # log.info("future_0_throughput = %s", df["future_0_throughput"].values[0])
    # log.info("future_1_throughput = %s", df["future_1_throughput"].values[0])
    # log.info("future_2_throughput = %s", df["future_2_throughput"].values[0])

    # 4. Find the GC-End
    df, n_slow_io = find_gc_end(df, median_throughput=median_throughput, n_future=N_FUTURE)
    log.info("n_slow_io = %s", n_slow_io)

    # 5. Filter outlier
    if filter_outlier:
        log.info("Filtering outlier")

        df, n_outlier1, n_outlier2, n_outlier3 = filter_outlier(df, median_latency, median_throughput, ip_latency_threshold)

        log.info("Outlier within slow period = %s", n_outlier1)
        log.info("Outlier within fast period = %s", n_outlier2)
        log.info("Outlier short tail spike = %s", n_outlier3)

    # 6. Write the result
    dropped_columns = ["sep", "sep2"]
    for i in range(N_HISTORY):
        dropped_columns.append(f"hist_{i}_throughput")
        dropped_columns.append(f"hist_{i}_latency")
    dropped_columns.append("throughput_drop")
    for i in range(N_FUTURE):
        dropped_columns.append(f"future_{i}_throughput")
    # dropped_columns = ["sep", "sep2", "n_hist_throughput", "n_hist_latency", "n_future_throughput"]
    df = df.drop(dropped_columns, axis=1, errors="ignore")
    df = df.reset_index(drop=True)

    stats_n_labeled = len(df)
    log.info("#IO labeled = " + str(stats_n_labeled))

    # 7. Write labeled data
    important_columns = ["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "size_after_replay", "mark_tail"]
    df = df.loc[:, df.columns.intersection(important_columns)]
    df["reject"] = df.apply(lambda row: 1 if (row["mark_tail"] == " Tail-Period ") else 0, axis=1)

    df = df.drop("mark_tail", axis=1, errors="ignore")

    stats_n_fast_io = len(df[df["reject"] == 0])
    stats_n_slow_io = len(df[df["reject"] == 1])

    log.info("Fast IO = " + str(stats_n_fast_io))
    log.info("Slow IO = " + str(stats_n_slow_io))

    final_df = df.copy()

    # 9. Filter out the Write IOs
    df = df[df["io_type"] == 1]  # Only check the read
    stats_n_read_io_labeled = len(df)
    stats_n_fast_read_io = len(df[df["reject"] == 0])
    stats_n_slow_read_io = len(df[df["reject"] == 1])

    # 10. Calculate the CDF gain on Read-only IO
    y_best = df.loc[df["reject"] == 0, "latency"]
    y_raw = df["latency"].values
    stats_cdf_gain = calc_cdf_gain(y_raw, y_best)

    # 11. Log the stats
    log.info("============================================")
    log.info("                BASIC INFO ", tab=16, tab_char=" ")
    log.info("============================================")
    stats_read_ratio = int((stats_n_read / stats_total_io) * 100)
    log.info("R:W ratio    = %s:%s", stats_read_ratio, 100 - stats_read_ratio)
    log.info("#IO          = %s", stats_total_io)
    log.info("#writes      = %s", stats_total_io - stats_n_read)
    log.info("#reads       = %s", stats_n_read)
    log.info("============================================")
    log.info("                STATISTICS", tab=16, tab_char=" ")
    log.info("============================================")
    log.info("IP latency                 = %s us (%s%%)", ip_latency_threshold, round(ip_latency_percent, 2))
    log.info("IP throughput              = %s B/us (%s%%)", ip_throughput_threshold, round(ip_thpt_percent, 2))
    log.info("Median latency             = %s us (50%%)", ip_latency_threshold)
    log.info("Median throughput          = %s B/us (50%%)", ip_latency_threshold)
    if filter_outlier:
        log.info("Outlier within slow period = %s (%s)", n_outlier1, calc_percent(n_outlier1, stats_total_io))
        log.info("Outlier within fast period = %s (%s)", n_outlier2, calc_percent(n_outlier2, stats_total_io))
        log.info("Outlier short tail spike   = %s (%s)", n_outlier3, calc_percent(n_outlier3, stats_total_io))
        stats_n_outlier = n_outlier3 + n_outlier2 + n_outlier1
        stats_percent_outlier = calc_percent(stats_n_outlier, stats_total_io)
        log.info("#Outlier IO                = %s (%s out of %s)", stats_n_outlier, stats_percent_outlier, stats_total_io)
    log.info("#IO labeled                = %s (%s out of %s)", stats_n_labeled, calc_percent(stats_n_labeled, stats_total_io), stats_total_io)
    log.info("#Write IO                  = %s", stats_n_labeled - stats_n_read_io_labeled)
    log.info("#Read IO                   = %s", stats_n_read_io_labeled)
    log.info("Fast R/W IO                = %s (%s out of %s)", stats_n_fast_io, calc_percent(stats_n_fast_io, stats_n_labeled), stats_n_labeled)
    log.info("Slow R/W IO                = %s (%s out of %s)", stats_n_slow_io, calc_percent(stats_n_slow_io, stats_n_labeled), stats_n_labeled)
    stats_percent_fast_read = calc_percent(stats_n_fast_read_io, stats_n_read_io_labeled, 0)
    stats_percent_slow_read = calc_percent(stats_n_slow_read_io, stats_n_read_io_labeled, 0)
    log.info("Fast Read-IO               = %s (%s out of %s)", stats_n_fast_read_io, stats_percent_fast_read, stats_n_read_io_labeled)
    log.info("Slow Read-IO               = %s (%s out of %s)", stats_n_slow_read_io, stats_percent_slow_read, stats_n_read_io_labeled)
    log.info("CDF gain                   = %s", stats_cdf_gain)

    return final_df
