import numpy as np
import pandas as pd

#  if calculate_agg:
#         log.info("[FEv6TS] Calculating rolling agg")
#         df["norm_ts_record"] = df["ts_record"] - df["ts_record"].min()
#         df = _calculate_agg(df, "norm_ts_record", N_WINDOW)


def calculate_agg(data: pd.DataFrame, group_col: str, window_size: int = 10):
    """
    Calculate the rolling average of IOPS for each 'ts_record' change in 'df' grouped by 'group_col',
    with a window size of 2 'ts_record' changes, falling back to 1 if necessary.

    :param df: DataFrame to process.
    :param group_col: The name of the column to group by.
    :return: DataFrame with the rolling average IOPS in a new column 'rolling_avg_iops'.
    """
    df = data.copy()

    # Group by 'group_col' and calculate the size and time difference
    grouped = df.groupby(group_col)
    agg_operations = grouped.size()
    time_diff = grouped[group_col].apply(lambda x: x.diff().mean()).fillna(0) / 1000  # time diff in seconds, mean diff for each group

    # Calculate IOPS for each group
    iops = agg_operations.shift(1) / time_diff.replace(0, 0.001)  # avoid division by zero
    iops.fillna(0, inplace=True)  # handle NaN

    # calculate num read for each group
    num_read = grouped["io_type"].apply(lambda x: (x == 1).sum()).shift(1).fillna(0)
    # replace num_read 0 to 0.001 to avoid division by zero
    num_read.replace(0, 0.001, inplace=True)
    num_write = grouped["io_type"].apply(lambda x: (x == 0).sum()).shift(1).fillna(0)
    # replace num_write 0 to 0.001 to avoid division by zero
    num_write.replace(0, 0.001, inplace=True)
    rw_ratio = num_read / num_write
    rw_ratio.replace(np.inf, -1, inplace=True)
    rw_ratio.fillna(-1, inplace=True)

    # Prepare a DataFrame for rolling computation
    agg_df = pd.DataFrame(
        {
            "ts_record": agg_operations.index,
            "iops": iops.values,
            "num_read": num_read.values,
            "num_write": num_write.values,
            "rw_ratio": rw_ratio.values,
            "iat": time_diff.values,
        }
    )
    # remove the first ts_record
    agg_df = agg_df[agg_df["ts_record"] != agg_df["ts_record"].min()]

    # Calculate rolling average of IOPS respecting ts_record changes
    # rolling_agg = agg_df["iops"].rolling(window=2, min_periods=1).mean()
    rolling_agg = agg_df[[col for col in agg_df.columns if col != "ts_record"]].rolling(window=window_size, min_periods=1).mean()

    # Create a mapping from ts_record to rolling_agg
    for col in rolling_agg.columns:
        rolling_map = pd.Series(rolling_agg[col].values, index=agg_df["ts_record"]).to_dict()
        df[f"{col}"] = df[group_col].map(rolling_map).fillna(0)

    return df


__all__ = ["calculate_agg"]
