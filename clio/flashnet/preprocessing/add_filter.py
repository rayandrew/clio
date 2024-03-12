#!/usr/bin/env python3

import argparse
import bisect
import math
import os
import subprocess
import sys
from pathlib import Path
from subprocess import call

import numpy as np
import pandas as pd
from sklearn.metrics import auc

import matplotlib.pyplot as plt

import clio.flashnet.ip_finder as ip_finder
from clio.utils.logging import log_get

log = log_get(__name__)

# These all numbers should be the same [5, 4]
N_HISTORY = 3
N_FUTURE = 3

# Filtering slow IO
THPT_DROP_RATE = 1.7


def read_file(input_file: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_file, sep=",")
    # # Make sure it has 7 columns
    # assert 7 == df.shape[1]
    # Rename column
    # Format = ts_record(ms),latency(us),io_type(r=1/w=0),
    #          size(B),offset,ts_submit(ms),size_after_replay(B)
    df.columns = [
        "ts_record",
        "size",
        "queue_len",
        "prev_queue_len_1",
        "prev_queue_len_2",
        "prev_queue_len_3",
        "prev_latency_1",
        "prev_latency_2",
        "prev_latency_3",
        "prev_throughput_1",
        "prev_throughput_2",
        "prev_throughput_3",
        "latency",
        "reject",
    ]

    # filter: remove io that doesn't executed properly (can't read/write all bytes)
    # df = df[df['size'] == df['size_after_replay']]
    return df


def add_filter(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # 1. Add more variable to Analyze the Trace
    df["throughput"] = df["size"] / df["latency"]  # add throughput column

    # 2. Find SLOW throughput and Get the Latency Threshold
    ip_latency_threshold, ip_latency_percent = ip_finder.tangent_based(df["latency"])
    # if throughput is higher, it is definitely FAST IO
    ip_throughput_threshold, ip_thpt_percent = ip_finder.tangent_based(df["throughput"])

    if ip_latency_percent < 50:
        log.error("ERROR: this trace profile is BAD because the IP latency is < 50%. Flashnet won't be able to make any significant improvement.")

    # slow throughput if it is less than the median throughput
    median_throughput = np.percentile(df["throughput"], 50)
    # if less than median_latency, the IO is FAST
    median_latency = np.percentile(df["latency"], 50)
    log.info("IP latency threshold : %s (%s%%)", ip_latency_threshold, ip_latency_percent)
    log.info("Median throughput threshold : %s (%s%%)", median_throughput, 50)
    log.info("IP throughput threshold : %s (%s%%)", ip_throughput_threshold, ip_thpt_percent)
    log.info("Median latency threshold : %s (%s%%)", median_latency, 50)

    df["mark_tail"] = df.apply(lambda row: " Tail-Period " if (row["reject"] == 1) else "  ", axis=1)

    # 5. Mark outlier in between GC period
    # 5.1 Outlier = Latency under the median_latency
    df["mark_outlier"] = df.apply(lambda row: "  ", axis=1)

    max_idx = len(df) - 1
    n_outlier1 = 0
    n_outlier2 = 0
    idx = 0
    # Iterate the dataframe to mark the outlier
    while idx <= max_idx:
        row = df.iloc[idx]
        if row["mark_tail"] == " Tail-Period ":
            # SLOW IO category
            if row["latency"] <= median_latency and row["throughput"] >= median_throughput:
                # Fast IO should NOT be within the tail period
                df.at[idx, "mark_outlier"] = " outlier1 "
                n_outlier1 += 1
        else:
            # FAST IO category
            # Very slow IO should NOT be here
            if row["latency"] > ip_latency_threshold:
                # Check the throughput, maybe it is heavy (io_size is big)
                if row["throughput"] < median_throughput:
                    df.at[idx, "mark_outlier"] = " outlier2 "
                    n_outlier2 += 1
        idx += 1

    log.info("Outlier within slow period = %s", n_outlier1)
    log.info("Outlier within fast period = %s", n_outlier2)

    # Remove outlier
    df = df[df["mark_outlier"] == "  "]
    df = df.reset_index(drop=True)

    # 6. Remove Outlier spike
    # Must be done after removing outlier1 and outlier2
    # Remove tail that only has IO <= the N_HISTORY
    max_idx = len(df) - 1
    n_outlier3 = 0
    idx = 0
    # Iterate the dataframe
    while idx <= max_idx:
        row = df.iloc[idx]
        # Will start processing at " GC-Start " marker
        if row["mark_tail"] == " Tail-Period ":
            n_tail = 1
            # going down checking the next slow IOs
            while idx < max_idx:
                idx += 1
                row = df.iloc[idx]
                if row["mark_tail"] != " Tail-Period ":
                    if n_tail <= N_HISTORY:
                        # mark this period as outlier
                        n_outlier3 += n_tail
                        while n_tail > 0:
                            df.at[idx - n_tail, "mark_outlier"] = " outlier3 "
                            n_tail -= 1
                    break
                n_tail += 1
        idx += 1

    log.info("Outlier short tail spike = %s", n_outlier3)

    # Remove outlier
    df = df[df["mark_outlier"] == "  "]
    df = df.reset_index(drop=True)

    # 7. Write data as labeled dataset
    # drop unnecessary columns
    # important_columns = [
    #     "ts_record",
    #     "size",
    #     "queue_len",
    #     "prev_queue_len_1",
    #     "prev_queue_len_2",
    #     "prev_queue_len_3",
    #     "prev_latency_1",
    #     "prev_latency_2",
    #     "prev_latency_3",
    #     "prev_throughput_1",
    #     "prev_throughput_2",
    #     "prev_throughput_3",
    #     "latency",
    #     "reject",
    #     "mark_tail",
    # ]
    # df = df.loc[:, df.columns.intersection(important_columns)]
    df["reject"] = df.apply(lambda row: 1 if (row["mark_tail"] == " Tail-Period ") else 0, axis=1)

    # drop marker column
    df = df.drop("mark_tail", axis=1)
    # df = df[
    #     [
    #         "ts_record",
    #         "size",
    #         "queue_len",
    #         "prev_queue_len_1",
    #         "prev_queue_len_2",
    #         "prev_queue_len_3",
    #         "prev_latency_1",
    #         "prev_latency_2",
    #         "prev_latency_3",
    #         "prev_throughput_1",
    #         "prev_throughput_2",
    #         "prev_throughput_3",
    #         "latency",
    #         "reject",
    #     ]
    # ]

    stats_n_fast_io = len(df[df["reject"] == 0])
    stats_n_slow_io = len(df[df["reject"] == 1])
    log.info("Fast IO = %s", stats_n_fast_io)
    log.info("Slow IO = %s", stats_n_slow_io)

    return df


def add_filter_v2(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["throughput"] = df["size"] / df["latency"]  # add throughput column

    # 2. Find SLOW throughput and Get the Latency Threshold
    ip_latency_threshold, ip_latency_percent = ip_finder.tangent_based(df["latency"])
    # if throughput is higher, it is definitely FAST IO
    ip_throughput_threshold, ip_thpt_percent = ip_finder.tangent_based(df["throughput"])

    if ip_latency_percent < 50:
        log.error("ERROR: this trace profile is BAD because the IP latency is < 50%. Flashnet won't be able to make any significant improvement.")

    # slow throughput if it is less than the median throughput
    median_throughput = np.percentile(df["throughput"], 50)
    # if less than median_latency, the IO is FAST
    median_latency = np.percentile(df["latency"], 50)
    log.info("IP latency threshold : %s (%s%%)", ip_latency_threshold, ip_latency_percent)
    log.info("Median throughput threshold : %s (%s%%)", median_throughput, 50)
    log.info("IP throughput threshold : %s (%s%%)", ip_throughput_threshold, ip_thpt_percent)
    log.info("Median latency threshold : %s (%s%%)", median_latency, 50)

    df["mark_tail"] = df.apply(lambda row: " Tail-Period " if (row["reject"] == 1) else "  ", axis=1)

    # 5. Mark outlier in between GC period
    # 5.1 Outlier = Latency under the median_latency
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

    log.info("Outlier within slow period = %s", n_outlier1)
    log.info("Outlier within fast period = %s", n_outlier2)

    # =================================================================================================

    # Identify tail periods
    # df["is_tail_period"] = df["mark_tail"] == " Tail-Period "

    # # Find consecutive tail groups
    # df["tail_group"] = df["is_tail_period"].ne(df["is_tail_period"].shift()).cumsum()

    # # Filter to get tail groups only
    # tail_groups = df[df["is_tail_period"]].groupby("tail_group").size().reset_index(name="count")

    # # Identify groups that are outliers
    # outlier_groups = tail_groups[tail_groups["count"] <= N_HISTORY]["tail_group"]

    # # edge cases the last group might not have enough data to be considered as an outlier

    # # Mark outliers in the original dataframe
    # df.loc[df["tail_group"].isin(outlier_groups), "mark_outlier"] = " outlier3 "
    # n_outlier3 = df["mark_outlier"].eq(" outlier3 ").sum()

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

    # # Log the count of 'Short Tail Spikes'
    log.info("Outlier short tail spike = %s", n_outlier3)

    df = df[df["mark_outlier"] == "  "]
    df = df.reset_index(drop=True)

    df = df.drop(["mark_tail", "is_tail_period", "tail_group", "mark_outlier"], axis=1, errors="ignore")

    return df


__all__ = ["add_filter"]
