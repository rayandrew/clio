#!/usr/bin/env python3

import bisect

import numpy as np
import pandas as pd
from sklearn.metrics import auc

import clio.flashnet.ip_finder as ip_finder
from clio.flashnet.constants import N_FUTURE, N_HISTORY, THPT_DROP_RATE

from clio.utils.logging import log_get

log = log_get(__name__)


# FEATURE 1: add N last latency
def collect_history(df: pd.DataFrame, col_name: str, n_history: int = N_HISTORY):
    # log.info(len(df))
    history_holder = pd.DataFrame()
    for n in range(1, n_history + 1):
        # get the history (adding 0 for the first IOs that doesn't have)
        values = ([0] * n) + list(df[col_name].values)
        values = values[: len(values) - n]  # remove extra value
        history_holder[n] = values
    history_holder["all_history"] = history_holder.values.tolist()
    return history_holder["all_history"].values


# FEATURE 1: add N future entry (The current IO is regarded as part of the future)
def collect_future(df: pd.DataFrame, col_name: str, n_future: int = N_FUTURE):
    # log.info(len(df))
    future_holder = pd.DataFrame()
    for n in range(0, n_future):
        # get the future (adding 0 for the last IOs that doesn't have)
        values = list(df[col_name].values) + ([0] * n)
        values = values[n : len(values)]  # remove extra value
        future_holder[n] = values
    future_holder["all_future"] = future_holder.values.tolist()
    return future_holder["all_future"].values


def mark_possible_start_1(row: pd.Series, ip_latency_threshold: float, ip_throughput_threshold: float, thpt_drop_rate: float):
    if row["throughput"] > ip_throughput_threshold or row["latency"] < ip_latency_threshold:
        # This IO is definitely fast enough, thus, can't be the GCstart
        return " "
    else:
        if row["throughput_drop"] >= thpt_drop_rate:
            return " GC-Start1 "
        else:
            return " "


def calc_percent(partition: float, total: float, precision: int = 2):
    return str(round(partition * 100 / total, precision)) + "%"


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
    log.info(idx)
    x_1 = x_1[:idx]
    y_1 = y_1[:idx]

    # Must add padding to make sure that it reach the x limit
    if max(x_1) < max_tail_value:
        x_1 = np.append(x_1, max_tail_value)
        y_1 = np.append(y_1, 1)

    # log.info(len(y_raw), len(x_2))
    # log.info(len(y_best), len(x_1), len(y_1))

    # 3. Calculate the AUC
    cdf_raw = auc(x_2, y_2)
    cdf_best = auc(x_1, y_1)

    # log.info('Raw CDF area  : {}'.format(cdf_raw))
    # log.info('Best CDF area : {}'.format(cdf_best))
    percent_gain = calc_percent(cdf_best - cdf_raw, cdf_raw)
    # log.info(percent_gain)

    # plt.figure(figsize=(7,3))
    # plt.xlabel('Latency (us)')
    # plt.ylabel('CDF')
    # plt.xlim(0, x_limiter) # Hopefully the x axis limit can catch the tail
    # # plt.ylim(0, 1)
    # plt.plot(x_2, y_2, label = "Raw latency", color="red")
    # plt.plot(x_1, y_1, label = "FlashNet-best-case", color="green")
    # plt.legend(loc="lower right")
    # plt.savefig("figure_path.png", bbox_inches='tight')
    # log.info("===== output figure : " + "figure_path.png")
    # log.info(percent_gain)
    # exit(0)
    return percent_gain
    # Note: Do not use np.trapz(xx,yy), it doesn't calculate valid area


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
    # correction = 0.8 Means that, we reduce the IP percentile by 0.2 or 20%
    # slow_throughput_threshold = np.percentile(df['throughput'], ip_thpt_percent) # * THPT_IP_CORRECTION
    # log.info ("   true percentile = " + str(ip_thpt_percent) + "; corrected slow_throughput_threshold = " + str(slow_throughput_threshold))

    # 3. Find the GC-Start
    df["n_hist_throughput"] = collect_history(df, "throughput", n_history=N_HISTORY)
    df["n_hist_latency"] = collect_history(df, "latency", n_history=N_HISTORY)

    # Check based on the current vs previous latency
    # df["latency_increase"] = df.apply (lambda row: round(row['latency'] / (row['n_hist_latency'][0] + 1), 1) , axis=1)

    # Check based on the current vs previous throughput
    df["throughput_drop"] = df.apply(lambda row: round(row["n_hist_throughput"][0] / (row["throughput"] + 0.1), 1), axis=1)
    log.info("Throughput drop %s", df["throughput_drop"].mean())

    # DAN: IMPORTANT VARIABLE for tuning
    # lat_increase_rate = LAT_INCREASE_RATE   # lower than this, it is not in the GC region
    thpt_drop_rate = THPT_DROP_RATE  # lower than this, it is not in the GC region
    # analyze the latency_increase and throughput_drop;
    # will also use ip_latency; the gc-start should be higher than the ip_latency
    df["mark_start1"] = df.apply(lambda row: mark_possible_start_1(row, ip_latency_threshold, ip_throughput_threshold, thpt_drop_rate), axis=1)
    # log.info("GC start %s", df["mark_start1"].value_counts())

    # # Check based on next throughput, the next should be 2x larger!
    df["n_future_throughput"] = collect_future(df, "throughput", n_future=N_FUTURE)

    # 4. Find the GC-END
    # GC-END = If N_FUTURE (3) consecutive IOs has throughput higher than median
    n_slow_io = 0
    # Iterate starting at GC-Start1
    df["mark_gc_end"] = df.apply(lambda row: "  ", axis=1)
    max_idx = len(df) - 1
    idx = 0
    # Iterate the dataframe
    while idx <= max_idx:
        row = df.iloc[idx]
        # Will start processing at " GC-Start " marker
        if row["mark_start1"] == " GC-Start1 ":
            n_slow_io += 1
            df.at[idx, "mark_tail"] = " Tail-Period "  # Mark the START
            # going down checking the future thpt
            while idx < max_idx:
                idx += 1
                row = df.iloc[idx]
                # idxs.append(idx)
                if all(i >= median_throughput for i in row["n_future_throughput"]):
                    # if Yes, it is the GC-END; no need to mark anything
                    break
                else:
                    n_slow_io += 1
                    # idxs.append(idx)
                    df.at[idx, "mark_tail"] = " Tail-Period "  # Mark it as slow
        # check next row until finding the starting point of the GC
        idx += 1
    log.info("n_slow_io = %s", n_slow_io)

    if filter_outlier:
        log.info("Filtering the outlier")

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
        log.info("Outlier short tail spike = " + str(n_outlier3))

        # Remove outlier
        df = df[df["mark_outlier"] == "  "]

    df = df.reset_index(drop=True)

    # 7. Write the marked data
    df = df.drop("n_hist_throughput", axis=1)
    df = df.drop("n_hist_latency", axis=1)
    df = df.drop("n_future_throughput", axis=1)
    stats_n_labeled = len(df)
    log.info("#IO labeled = " + str(stats_n_labeled))

    # write_to_file(latency_df, outfile_path + ".tmp", True)

    # 8. Write data as labeled dataset
    # drop unnecessary columns
    important_columns = ["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "size_after_replay", "mark_tail"]
    df = df.loc[:, df.columns.intersection(important_columns)]
    df["reject"] = df.apply(lambda row: 1 if (row["mark_tail"] == " Tail-Period ") else 0, axis=1)

    # drop marker column
    df = df.drop("mark_tail", axis=1)

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
    log.info("                BASIC INFO ")
    log.info("============================================")
    stats_read_ratio = int((stats_n_read / stats_total_io) * 100)
    log.info("R:W ratio    = %s:%s", stats_read_ratio, 100 - stats_read_ratio)
    log.info("#IO          = %s", stats_total_io)
    log.info("#writes      = %s", stats_total_io - stats_n_read)
    log.info("#reads       = %s", stats_n_read)
    log.info("============================================")
    log.info("                STATISTICS")
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


__all__ = ["labeling"]
