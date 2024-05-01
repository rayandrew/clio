#!/usr/bin/env python3

# based on feat_v6_ts.py
# adopted because we need to modify the existing post-feature-engineering dataset


import numpy as np
import pandas as pd

from clio.utils.logging import log_get

N_HISTORY = 3
N_WINDOW = 10
log = log_get(__name__)


# def append_prev_feature(df: pd.DataFrame, num: int, colname: str):
#     for i in range(1, num + 1):
#         df["prev_" + colname + "_" + str(i)] = df[colname].shift(i).values


def append_prev_feature(df, num, colname, prev_df=None):
    count_df = df.shape[0]
    if prev_df is not None:
        concat_df = pd.concat([prev_df.iloc[-num:], df], axis=0)
        assert concat_df.shape[0] == num + count_df
        for i in range(1, num + 1):
            # this shift operation might be NaN if the first few IOs don't have enough historical data
            # so we used the last valid value from prev_df instead
            concat_df["prev_" + colname + "_" + str(i)] = concat_df[colname].shift(i).values

        # we remove the first few IOs that comes from prev_df
        concat_df = concat_df.iloc[num:]
        for i in range(1, num + 1):
            df["prev_" + colname + "_" + str(i)] = concat_df["prev_" + colname + "_" + str(i)].values

    else:
        for i in range(1, num + 1):
            # this shift operation might be NaN if the first few IOs don't have enough historical data
            # so we used the last valid value from prev_df instead
            df["prev_" + colname + "_" + str(i)] = df[colname].shift(i).values


def append_queue_len(latency, ts_submit):
    queue_process = []
    queue_len = []
    for i in range(len(ts_submit)):
        while queue_process and queue_process[0] < ts_submit[i]:
            queue_process.pop(0)
        queue_process.append(ts_submit[i])
        queue_process.sort()
        queue_len.append(len(queue_process))
    return queue_len


def feature_engineering(data: pd.DataFrame, prev_data: pd.DataFrame | None = None):
    # Assumption: prev data here is the previous data that has been feature engineered

    df = data.copy()
    #    ts_record  latency  io_type   size    offset  ts_submit  size_after_replay  reject
    df["queue_len"] = append_queue_len(df["latency"].tolist(), df["ts_submit"].tolist())

    # Drop unnecessary columns
    # cols = ["offset", "ts_submit", "size_after_replay"]
    # df = df.drop(columns=cols, axis=1, errors="ignore")

    # Calculate per-IO throughput
    df["throughput"] = df["size"] / df["latency"]
    df["throughput"] = df["throughput"].round(0)

    if prev_data is not None:
        # ts_record should be normalized to 0 within chunk here
        df["ts_record"] = df["ts_record"] + prev_data["ts_record"].max()
        prev_data["throughput"] = prev_data["size"] / prev_data["latency"]
        prev_data["throughput"] = prev_data["throughput"].round(0)

    # Append Historical data
    append_prev_feature(df, N_HISTORY, "queue_len", prev_df=prev_data)
    append_prev_feature(df, N_HISTORY, "latency", prev_df=prev_data)
    append_prev_feature(df, N_HISTORY, "throughput", prev_df=prev_data)

    # Drop the first few IOs that don't have a complete historical data
    if prev_data is None:
        df.drop(df.head(N_HISTORY).index, inplace=True)
        log.info("[FEv6TS] Removed %d first IOs because they don't have enough historical data", N_HISTORY)

    # Calculate latency increase
    df["latency_increase"] = (df["latency"] / df["prev_latency_1"]).round(2)

    # Calculate throughput drop
    df["throughput_drop"] = (df["prev_throughput_1"] / (df["throughput"] + 0.1)).round(2)

    # Remove any latency-related feature, except the historical value and the "latency" column
    # The latency column is needed for drawing CDF latency, not to be used as input feature
    df = df.drop(columns=["throughput", "throughput_drop", "latency_increase"], axis=1)

    # Put non_input_feature column at the last
    non_input_feature = ["latency", "reject"]
    input_features = [col for col in df.columns if col not in non_input_feature]
    df = df[input_features + non_input_feature]

    # Label all Write IO as non rejectable
    log.info("[FEv6TS] Labeling all Write IO as non rejectable")
    df.loc[df["io_type"] == 0, "reject"] = 0

    # Create dataset without Write IO
    readonly_df = df.copy()
    readonly_df = readonly_df[readonly_df["io_type"] == 1]
    readonly_df = readonly_df.drop(["io_type"], axis=1)
    df = df.reset_index(drop=True)
    readonly_df = readonly_df.reset_index(drop=True)
    return df, readonly_df


__all__ = ["feature_engineering"]
