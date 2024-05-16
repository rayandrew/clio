from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator

import pandas as pd

from clio.utils.logging import log_get

log = log_get(__name__)


def normalize_df_ts_record(df: pd.DataFrame):
    # get min ts_record
    df = df.copy()
    ts_record_min = df["ts_record"].min()
    df["ts_record"] = df["ts_record"] - ts_record_min
    return df


@dataclass(kw_only=True)
class TraceWindowGeneratorContext:
    total_processed_ios: int = 0
    last_ts_record: int = 0


def trace_time_window_generator(
    ctx: TraceWindowGeneratorContext,
    window_size: int | float,
    current_trace: pd.DataFrame,
    trace_paths: list[Path],
    n_data: int,
    reference: pd.DataFrame,
    curr_count: int = 0,
    end_ts: int = -1,
    curr_ts_record: float = 0.0,
    return_last_remaining_data: bool = False,
    ts_offset: float = 0.0,
    query: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    reader: Callable[[str | Path], pd.DataFrame] = pd.read_csv,
) -> Generator[tuple[int, TraceWindowGeneratorContext, pd.DataFrame, pd.DataFrame, bool, bool], None, None]:
    """Generate time-based window

    Args:
        ctx (TraceWindowGeneratorContext): Window generator context
            ctx.window_size is in seconds
        total_data_needed (int): Total data needed
        current_trace (pd.DataFrame): Current trace dataframe
        trace_paths (List[Path]): List of trace paths
        n_data (int): Number of dataframes
        reference_data (pd.DataFrame): Reference dataframe
        curr_count (int, optional): Current dataframe count. Defaults to 0.
        end_ts (int, optional): End timestamp. Defaults to -1.
        curr_ts_record (float, optional): Current timestamp record. Defaults to 0.0
        return_last_remaining_data (bool, optional): Return last remaining data. Defaults to False.
        ts_offset (int, optional): Timestamp offset. Defaults to 0.
    """
    # NOTE: ts_record IS IN SECONDS

    assert window_size > 0, "window_size must be > 0 and in seconds"

    interval = pd.DataFrame()
    last = False
    window_size_ms = window_size * 1000
    window_count = 0

    while True:
        if end_ts > 0 and curr_ts_record >= end_ts:
            log.debug("Stopping due to end_ts (%d) reached", end_ts, tab=1)
            break

        log.debug("Start ts record: %d", curr_ts_record, tab=1)
        log.debug("End ts record: %d", curr_ts_record + window_size_ms, tab=1)
        picked = current_trace[(current_trace["ts_record"] >= curr_ts_record) & (current_trace["ts_record"] < curr_ts_record + window_size_ms)]
        log.debug("Picked size: %d", len(picked), tab=1)
        n_picked = len(picked)

        if n_picked == 0:
            if last:
                log.debug("End of data", tab=1)
                log.debug("Reference size: %d, will be skipped", len(reference), tab=2)
                log.debug("Interval size: %d, will be skipped", len(interval), tab=2)

                log.debug("Reaching the last trace", tab=2)
                if return_last_remaining_data and len(interval) > 0:
                    # ctx.total_processed_ios += len(interval)
                    window: pd.DataFrame = interval.copy()  # type: ignore
                    if len(window) > 0:
                        interval_s = (window["ts_record"].iloc[-1] - window["ts_record"].iloc[0]) / (1000)  # type: ignore
                    else:
                        interval_s = 0.0
                    is_interval_valid = abs(interval_s - window_size) <= 1e-1
                    window_count += 1
                    yield window_count, ctx, reference, window, is_interval_valid, True

                break
            else:
                curr_count += 1
                if curr_count >= n_data:
                    last = True

                if not last:
                    current_trace = reader(trace_paths[curr_count])
                    if query:
                        current_trace = current_trace[query(current_trace)]  # type: ignore
                    current_trace["original_ts_record"] = current_trace["ts_record"]
                    # apply timestamp offset
                    current_trace["ts_record"] += ts_offset

                continue

        picked = picked.copy(deep=True).reset_index(drop=True)
        interval = pd.concat([interval, picked])  # type: ignore
        interval_s = (interval["ts_record"].iloc[-1] - interval["ts_record"].iloc[0]) / (1000)  # type: ignore
        if interval_s > window_size:
            interval = interval[interval["ts_record"] >= (interval["ts_record"].iloc[-1] - window_size_ms)]  # type: ignore
            interval_s = (interval["ts_record"].iloc[-1] - interval["ts_record"].iloc[0]) / (1000)  # type: ignore

        is_drift_window = abs(interval_s - window_size) <= 0.5

        if is_drift_window:
            window: pd.DataFrame = interval.copy()  # type: ignore
            ctx.last_ts_record = round(interval["ts_record"].iloc[-1] + 0.1, 1)  # type: ignore
            window_count += 1
            yield window_count, ctx, reference, window, True, False

            curr_ts_record = ctx.last_ts_record
            reference = interval.copy()  # type: ignore
            reference = reference.reset_index(drop=True)
            interval = pd.DataFrame()
        else:
            curr_ts_record = round(interval["ts_record"].iloc[-1] + 0.1, 1)  # type: ignore


# def get_dataset_paths(
#     data_path: Path,
#     profile_name: str = "provide_v1",
#     feat_name: str = "feat_v6_ts",
#     readonly_data: bool = True,
# ) -> dict[str, list[Path]]:
#     if not data_path.exists():
#         return []

#     data_paths: dict[str, list[Path]] = {}

#     # list all directory
#     for dir in data_path.iterdir():
#         glob_path = "**/%s.%s" % (profile_name, feat_name)
#         if readonly_data:
#             glob_path += ".readonly"
#         glob_path += ".dataset"
#         datapaths = list(dir.glob(glob_path))
#         datapaths.sort(key=lambda x: int(x.parent.name.split("_")[1]))
#         data_paths[dir.name] = datapaths

#     if len(data_paths) == 0:
#         return data_paths

#     return data_paths


def trace_get_dataset_paths(
    data_path: Path,
    profile_name: str = "profile_v1",
    feat_name: str = "feat_v6_ts",
    readonly_data: bool = True,
    sort_by: Callable[[Path], int] | None = None,
) -> list[Path]:
    if not data_path.exists():
        return []

    log.debug("Profile name: %s", profile_name)
    log.debug("Feature name: %s", feat_name)
    log.debug("Readonly data: %s", readonly_data)

    data_paths: list[Path] = []

    glob_path = "**/*%s.%s" % (profile_name, feat_name)
    if readonly_data:
        glob_path += ".readonly"
    glob_path += ".dataset"
    data_paths = list(data_path.glob(glob_path))

    if len(data_paths) == 0:
        return []

    # check if splitted by checking chunk_* in path
    splitted = False
    for path in data_paths:
        if "chunk_" in path.parent.name:
            splitted = True
            break

    if splitted:  # sort df_paths by chunk id (parent folder)
        data_paths.sort(key=lambda x: int(x.parent.name.split("_")[1]))
    else:
        if sort_by:
            data_paths.sort(key=sort_by)

    return data_paths


def trace_get_labeled_paths(
    data_path: Path,
    profile_name: str = "profile_v1",
) -> list[Path]:
    if not data_path.exists():
        return []

    data_paths: list[Path] = []

    glob_path = "**/*.profile_v1.feat_v6_ts.dataset"
    data_paths = list(data_path.glob(glob_path))

    if len(data_paths) == 0:
        return []

    splitted = False
    for path in data_paths:
        if "chunk_" in path.parent.name:
            splitted = True
            break

    if splitted:
        data_paths.sort(key=lambda x: int(x.parent.name.split("_")[1]))

    return data_paths


__all__ = [
    "trace_time_window_generator",
    "TraceWindowGeneratorContext",
    "trace_get_dataset_paths",
    "trace_get_labeled_paths",
    "normalize_df_ts_record",
]
