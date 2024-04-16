from pathlib import Path

import pandas as pd


def mult_normalize(
    df: pd.DataFrame,
    ignore_columns: list[str] = [],
) -> pd.DataFrame:
    ignore_cols = ignore_columns + [
        "name",
        # "num_io",
        "disks",
        "start_ts",
        "end_ts",
        "ts_unit",
        "size_unit",
        "duration",
    ]
    # normalize columns below using min-max normalization, treat min as 1 and other values are values / min
    norm_df = df.copy()
    for column in df.columns.difference(ignore_cols):
        # min_val = np.max([float(norm_df[column].min()), 1e-7])
        min_val = norm_df[column].min()
        # _log.info("Column: %s, min val %s", column, min_val, tab=1)
        norm_df[column] = norm_df[column] / min_val if min_val > 0 else 1
    return norm_df


def parse_list_file(list_file: Path):
    traces: dict[str, dict[str, str]] = {}
    key = None
    counter = 0
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                continue

            # remove # .... from line
            line = line.split("#")[0]
            line = line.strip()

            if line.startswith("!"):
                counter = 0
                key = line[1:]
                key = key.strip()
                if key not in traces:
                    traces[key] = {}
                continue

            if ":" in line:
                name, value = line.split(":")
                name = name.strip()
                value = value.strip()
                traces[key][name] = value
                continue

            if key is not None:
                traces[key][counter] = line
                counter += 1

    return traces


__all__ = ["parse_list_file"]
