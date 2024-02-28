from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator

import pandas as pd

from clio.utils.logging import log_get

log = log_get(__name__)


def read_raw_trace_as_df(path: Path | str) -> pd.DataFrame:
    """Read raw trace

    Args:
        path (Path | str): Path to the trace

    Returns:
        pd.DataFrame: Trace dataframe
    """
    log.debug("Reading trace: %s", path)
    return pd.read_csv(
        path,
        names=["ts_record", "disk_id", "offset", "io_size", "read"],
        delimiter=" ",
        dtype={"ts_record": float, "disk_id": str, "offset": int, "io_size": int, "read": bool},
    )


@dataclass(kw_only=True)
class TraceWindowGeneratorContext:
    total_processed_ios: int = 0
    last_ts_record: int = 0


def trace_time_window_generator(
    ctx: TraceWindowGeneratorContext,
    window_size: int | float,
    current_trace_df: pd.DataFrame,
    trace_paths: list[Path],
    n_dfs: int,
    reference_df: pd.DataFrame,
    curr_df_count: int = 0,
    end_ts: int = -1,
    curr_ts_record: float = 0.0,
    return_last_remaining_data: bool = False,
    ts_offset: float = 0.0,
    query: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
) -> Generator[tuple[int, TraceWindowGeneratorContext, pd.DataFrame, pd.DataFrame, bool, bool], None, None]:
    """Generate time-based window

    Args:
        ctx (TraceWindowGeneratorContext): Window generator context
            ctx.window_size is in seconds
        total_data_needed (int): Total data needed
        current_trace_df (pd.DataFrame): Current trace dataframe
        trace_paths (List[Path]): List of trace paths
        n_dfs (int): Number of dataframes
        reference_df (pd.DataFrame): Reference dataframe
        curr_df_count (int, optional): Current dataframe count. Defaults to 0.
        end_ts (int, optional): End timestamp. Defaults to -1.
        curr_ts_record (float, optional): Current timestamp record. Defaults to 0.0
        return_last_remaining_data (bool, optional): Return last remaining data. Defaults to False.
        ts_offset (int, optional): Timestamp offset. Defaults to 0.
    """
    # NOTE: ts_record IS IN MILLISECONDS

    assert window_size > 0, "window_size must be > 0 and in seconds"

    interval_df = pd.DataFrame()
    last_df = False
    window_size_ms = window_size * 1000
    window_count = 0

    while True:
        if end_ts > 0 and curr_ts_record >= end_ts:
            log.debug("Stopping due to end_ts (%d) reached", end_ts, tab=1)
            break

        log.debug("Start ts record: %d", curr_ts_record, tab=1)
        log.debug("End ts record: %d", curr_ts_record + window_size_ms, tab=1)
        df_picked = current_trace_df[(current_trace_df["ts_record"] >= curr_ts_record) & (current_trace_df["ts_record"] < curr_ts_record + window_size_ms)]
        log.debug("Picked df size: %d", len(df_picked), tab=1)
        n_df_picked = len(df_picked)

        if n_df_picked == 0:
            if last_df:
                log.debug("End of data", tab=1)
                log.debug("Reference df size: %d, will be skipped", len(reference_df), tab=2)
                log.debug("Interval df size: %d, will be skipped", len(interval_df), tab=2)

                log.debug("Reaching the last trace", tab=2)
                if return_last_remaining_data and len(interval_df) > 0:
                    # ctx.total_processed_ios += len(interval_df)
                    window_df: pd.DataFrame = interval_df.copy()  # type: ignore
                    if len(window_df) > 0:
                        interval_s = (window_df["ts_record"].iloc[-1] - window_df["ts_record"].iloc[0]) / (1000)  # type: ignore
                    else:
                        interval_s = 0.0
                    is_interval_valid = abs(interval_s - window_size) <= 1e-1
                    window_count += 1
                    yield window_count, ctx, reference_df, window_df, is_interval_valid, True

                break
            else:
                curr_df_count += 1
                if curr_df_count >= n_dfs:
                    last_df = True

                if not last_df:
                    current_trace_df = read_raw_trace_as_df(trace_paths[curr_df_count])
                    current_trace_df = current_trace_df[query(current_trace_df)]  # type: ignore
                    current_trace_df["original_ts_record"] = current_trace_df["ts_record"]
                    # apply timestamp offset
                    current_trace_df["ts_record"] += ts_offset

                continue

        df_picked = df_picked.copy(deep=True).reset_index(drop=True)
        interval_df = pd.concat([interval_df, df_picked])  # type: ignore
        interval_s = (interval_df["ts_record"].iloc[-1] - interval_df["ts_record"].iloc[0]) / (1000)  # type: ignore
        if interval_s > window_size:
            interval_df = interval_df[interval_df["ts_record"] >= (interval_df["ts_record"].iloc[-1] - window_size_ms)]  # type: ignore
            interval_s = (interval_df["ts_record"].iloc[-1] - interval_df["ts_record"].iloc[0]) / (1000)  # type: ignore

        is_drift_window = abs(interval_s - window_size) <= 0.5

        if is_drift_window:
            window_df: pd.DataFrame = interval_df.copy()  # type: ignore
            ctx.last_ts_record = round(interval_df["ts_record"].iloc[-1] + 0.1, 1)  # type: ignore
            window_count += 1
            yield window_count, ctx, reference_df, window_df, True, False

            curr_ts_record = ctx.last_ts_record
            reference_df = interval_df.copy()  # type: ignore
            reference_df = reference_df.reset_index(drop=True)
            interval_df = pd.DataFrame()
        else:
            curr_ts_record = round(interval_df["ts_record"].iloc[-1] + 0.1, 1)  # type: ignore


__all__ = ["trace_time_window_generator", "TraceWindowGeneratorContext"]
