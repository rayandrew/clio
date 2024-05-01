#!/usr/bin/env python3

import numpy as np
import pandas as pd

import clio.flashnet.ip_finder as ip_finder
from clio.flashnet.preprocessing.labeling import do_filter_outlier

from clio.utils.logging import log_get

log = log_get(__name__)


def add_filter_v2(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["throughput"] = df["size"] / df["latency"]  # add throughput column

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
    log.info("IP latency threshold : %s (%s%%)", ip_latency_threshold, ip_latency_percent)
    log.info("Median throughput threshold : %s (%s%%)", median_throughput, 50)
    log.info("IP throughput threshold : %s (%s%%)", ip_throughput_threshold, ip_thpt_percent)
    log.info("Median latency threshold : %s (%s%%)", median_latency, 50)

    # df["mark_tail"] = df.apply(lambda row: " Tail-Period " if (row["reject"] == 1) else "  ", axis=1)
    df["mark_tail"] = np.where(df["reject"] == 1, " Tail-Period ", "  ")

    df, _n_outlier1, _n_outlier2, _n_outlier3 = do_filter_outlier(
        df, median_latency=median_latency, median_throughput=median_throughput, ip_latency_threshold=ip_latency_threshold
    )

    df = df.drop(["mark_tail", "is_tail_period", "tail_group", "mark_outlier"], axis=1, errors="ignore")

    return df


add_filter = add_filter_v2

__all__ = ["add_filter"]
