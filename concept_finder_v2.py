import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from scipy import stats

# from scipy.stats._stats_py import ModeResult

ModeResult = Dict[int, int]

@dataclass
class Row:
    index: int = -1
    value: float = 0.0
    stability: int = 0
    stability_group: int = 0
    group: int = -1
    prev_group: int = -1
    mode: ModeResult = field(default_factory=dict)
    next_mode: ModeResult = field(default_factory=dict)


STABILITY_THRESHOLD = 20
GROUP_THRESHOLD = 250
GROUP_OFFSET = 50
DRIFT_THRESHOLD = 50
ROLLING_WINDOW = 10


def determine_group(value: float, threshold: float, offset: float) -> int:
    if value < 0:
        return 0

    if value <= offset:
        return 1

    return int((value - offset) // threshold) + 2


def calc_mode_row(values: List[Row], key: str) -> ModeResult:
    mode_result: ModeResult = {}
    for value in values:
        val = getattr(value, key)
        if val not in mode_result:
            mode_result[val] = 0
        mode_result[val] += 1
    # sort mode_result by value
    mode_result = dict(sorted(mode_result.items(), key=lambda x: x[1], reverse=True))
    return mode_result

def lookback_nearest_drift_idx(rows: List[Row], drift_idx: int, key: str, criteria: int) -> int:
    new_drift_idx = drift_idx
    for i in range(new_drift_idx, -1, -1):
        if getattr(rows[i], key) == criteria:
            new_drift_idx = i
            break

    return new_drift_idx

def lookback_consecutive_drift_idx(rows: List[Row], drift_idx: int, key: str, criteria: int) -> int:
    # if criteria == getattr(rows[drift_idx], key):
    #     return drift_idx
    
    new_drift_idx = drift_idx
    for i, prev_i in zip(range(new_drift_idx, -1, -1), range(new_drift_idx - 1, -1, -1)):
        attr = getattr(rows[i], key)
        prev_attr = getattr(rows[prev_i], key)

        if attr != criteria and prev_attr == criteria:
            continue
        if attr == criteria and prev_attr != criteria:
            new_drift_idx = i
            break

    return new_drift_idx

def lookahead_nearest_drift_idx(rows: List[Row], drift_idx: int, key: str, criteria: int) -> int:
    new_drift_idx = drift_idx
    for i in range(new_drift_idx + 1, len(rows)):
        if getattr(rows[i], key) == criteria:
            new_drift_idx = i
            break

    return new_drift_idx

def lookahead_consecutive_drift_idx(rows: List[Row], drift_idx: int, key: str, criteria: int) -> int:
    # if criteria == getattr(rows[drift_idx], key):
    #     return drift_idx
    
    new_drift_idx = drift_idx
    for i, next_i in zip(range(new_drift_idx, len(rows)), range(new_drift_idx + 1, len(rows))):
        attr = getattr(rows[i], key)
        next_attr = getattr(rows[next_i], key)
        if attr != criteria and next_attr == criteria:
            continue

        if attr == criteria and next_attr != criteria:
            new_drift_idx = i
            break

    return new_drift_idx

def get_highest_mode(mode: ModeResult) -> int:
    return max(mode, key=mode.get) if mode else -1


def main():
    data_dir = Path("/home/cc/projects/clio/data-test")

    def rmdir(dir: Path):
        for p in dir.iterdir():
            if p.is_dir():
                rmdir(p)
            else:
                p.unlink()
        dir.rmdir()

    if data_dir.exists():
        rmdir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # with open("/home/cc/projects/clio/runs/raw/tencent/drift/1548/1m/iops/st-15.gt-250.go-50.dt-50.rw-10/raw.dat") as f:
    rows: List[Row] = []
    with open("/home/cc/projects/clio/raw.tsv") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            data = float(line)
            g = determine_group(data, threshold=GROUP_THRESHOLD, offset=GROUP_OFFSET)
            row = Row(index=i - 1, value=data, stability=0, stability_group=0, group=g, mode={}, next_mode={})
            prev_windows = rows[max(0, i - ROLLING_WINDOW) : i]
            prev_windows.append(row)
            row.mode = calc_mode_row(prev_windows, key="group")
            row.prev_group = rows[-1].group if rows else -1
            rows.append(row)

    num_data = len(rows)

    for i in range(1, num_data):
        row = rows[i]
        prev_row = rows[i - 1]
        # prev_windows = rows[max(0, i - ROLLING_WINDOW) : i]
        next_windows = rows[i : min(num_data, i + ROLLING_WINDOW)]
        row.next_mode = calc_mode_row(next_windows, key="group")

        curr_total_mode = sum(row.mode.values())
        curr_mode_group = row.mode.get(row.group, 0)
        next_mode_group = row.next_mode.get(row.group, 0)

        if curr_total_mode < ROLLING_WINDOW:
            row.stability = prev_row.stability + 1
        else:
            if curr_mode_group > (ROLLING_WINDOW / 2) and next_mode_group > (ROLLING_WINDOW / 2):
                row.stability = prev_row.stability + 1
            else:
                prev_group_mode = prev_row.mode.get(prev_row.group, 0)
                potential_modes = [row.mode.get(row.group + diff, 0) for diff in [-1, 1, 2, -1]]
                if prev_group_mode > (ROLLING_WINDOW / 2) or sum(potential_modes) > (ROLLING_WINDOW / 2):
                    row.stability = prev_row.stability + 1
                else:
                    row.stability = 0

        if row.group != prev_row.group:
            row.stability_group = 0
        else:
            row.stability_group = prev_row.stability_group + 1

    # create tsv file of row
    with open(data_dir / "row.tsv", "w") as f:
        f.write("index\tvalue\tstability\tstability_group\tgroup\tprev_group\tmode\tnext_mode\n")
        for i, row in enumerate(rows):
            f.write(
                f"{row.index}\t{row.value}\t{row.stability}\t{row.stability_group}\t{row.group}\t{row.prev_group}\t{row.mode}\t{row.next_mode}\n"
            )


    # find potential stable concepts
    potential_end_concepts: List[Row] = []
    for i, (c, n) in enumerate(zip(rows, rows[1:])):
        if c.stability >= STABILITY_THRESHOLD and n.stability == 0:
            assert c.index == i
            potential_end_concepts.append(c.index)

    possible_recurring_drifts: List[Tuple[int, int]] = []
    possible_other_drifts: List[Tuple[int, int]] = []

    print(potential_end_concepts)
    
    # concept finder
    # need to have 2 pointers, start and end
    possible_drifts: List[Tuple[int, int]] = []
    for i, start in enumerate(potential_end_concepts):
        for j in range(i + 1, len(potential_end_concepts)):
            end = potential_end_concepts[j]
            # if DRIFT_THRESHOLD < end - (start - rows[start].stability) < 3600:
            if end - (start - rows[start].stability) < 3600:
                possible_drifts.append((start, end))
            else:
                break

    # print(possible_drifts)
    drifts: List[Tuple[int, int, str]] = []
    recurring_drifts: List[Tuple[int, int]] = []

    for start_idx, end_idx in possible_drifts:
        # sys.exit(0)
        start_drift_idx = rows[start_idx].index - rows[start_idx].stability
        start = rows[start_drift_idx]
        end_drift_idx = end_idx
        end = rows[end_drift_idx]

        start_mode = get_highest_mode(start.mode)
        end_mode = get_highest_mode(end.mode)
        # print(start_idx, end_idx, "|", start_mode, end_mode)
        # if end_idx == 345:
        #         print("HIIII", rows[end_drift_idx])
        #         print("hi")


        if start_mode == end_mode:
            # recurring
            new_start_drift_idx = lookback_consecutive_drift_idx(rows=rows, drift_idx=start_drift_idx, key="group", criteria=start_mode)
            if rows[end_drift_idx].group != start_mode:
                new_end_drift_idx = lookback_nearest_drift_idx(rows=rows, drift_idx=end_drift_idx, key="group", criteria=start_mode)
            new_end_drift_idx = lookahead_consecutive_drift_idx(rows=rows, drift_idx=new_end_drift_idx, key="group", criteria=start_mode)
            # print("===", new_start_drift_idx, new_end_drift_idx, "recurring")
            # drifts.append((new_start_drift_idx, new_end_drift_idx, "recurring"))
            recurring_drifts.append((new_start_drift_idx, new_end_drift_idx))
        else:
            new_start_drift_idx = lookback_consecutive_drift_idx(rows=rows, drift_idx=start_drift_idx, key="group", criteria=start_mode)
            if rows[end_drift_idx].group != end_mode:
                new_end_drift_idx = lookback_nearest_drift_idx(rows=rows, drift_idx=end_drift_idx, key="group", criteria=end_mode)
            new_end_drift_idx = lookahead_consecutive_drift_idx(rows=rows, drift_idx=end_drift_idx, key="group", criteria=end_mode)

            diff_mode = abs(start_mode - end_mode)
            if diff_mode > 1:
                recurring_exists_within_range = False
                for recurring_start, recurring_end in recurring_drifts:
                    if recurring_start == new_start_drift_idx:
                        # print("recurring_start", recurring_start, new_start_drift_idx)
                        recurring_exists_within_range = True
                        break
                # print(recurring_exists_within_range)
                if not recurring_exists_within_range:
                    # print("===", new_start_drift_idx, new_end_drift_idx, "sudden")
                    drifts.append((new_start_drift_idx, new_end_drift_idx, "sudden"))
            else:
                ...
            
    # gradual_drifts: List[Tuple[int, int]] = []
    # for start_idx, end_idx in recurring_drifts:
    #     if i == 0:
    #         continue
    #     start = rows[start_idx]
    #     end = rows[end_idx]

    #     assert start.group == end.group

    #     for next_start_idx, next_end_idx in recurring_drifts:
    #         next_start = rows[next_start_idx]
    #         next_end = rows[next_end_idx]
    #         if start.group == next_start.group and end.group == next_end.group:
    #             break

    for start, end, type in drifts:
        print("==== %s %s %s" % (start, end, type))

if __name__ == "__main__":
    main()
