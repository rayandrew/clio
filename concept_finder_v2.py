import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

import pymannkendall as mk
from scipy import stats

# from scipy.stats._stats_py import ModeResult

ModeResult = Dict[int, int]

parser = ArgumentParser()
parser.add_argument("-i", type=int, default=1)
args = parser.parse_args()


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


# STABILITY_THRESHOLD = 20
# GROUP_THRESHOLD = 250
# GROUP_OFFSET = 50
# DRIFT_THRESHOLD = 50
# ROLLING_WINDOW = 10

STABILITY_THRESHOLD = 14
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
    # mode_result = dict(sorted(mode_result.items(), key=lambda x: x[1], reverse=True))
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


# def collapse_data(data: List[Tuple[int, int]], diff_threshold: int) -> List[Tuple[int, int]]:
def collapse_intervals(intervals: List[Tuple[int, int]], threshold: int) -> List[Tuple[int, int]]:
    if len(intervals) == 0:
        return []

    # Sort intervals by the start value
    intervals.sort(key=lambda x: x[0])
    collapsed: List[Tuple[int, int]] = [intervals[0]]

    for start, end in intervals[1:]:
        last_end = collapsed[-1][1]

        if start - last_end <= threshold:
            # Merge intervals
            collapsed[-1] = (collapsed[-1][0], end)
        else:
            # Add new interval to the result
            collapsed.append((start, end))

    return collapsed


def get_similar_modes(mode: ModeResult, criteria: int, diff: int = 2, include_self: bool = False) -> ModeResult:
    # if diff <= 0:
    #     raise ValueError("diff must be non-zero and non-negative")

    diffs = [0] if include_self else []
    if diff > 0:
        for i in range(1, diff + 1):
            diffs.append(i)
            diffs.append(-i)

    return {criteria + diff: mode.get(criteria + diff, 0) for diff in diffs}


def rmdir(dir: Path):
    if not dir.exists():
        return
    for p in dir.iterdir():
        if p.is_dir():
            rmdir(p)
        else:
            p.unlink()
    dir.rmdir()


def process(rows: List[Row], output_dir: Path):
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

    caches_modes: Dict[int, ModeResult] = {}

    for start_idx, end_idx in possible_drifts:
        # sys.exit(0)
        start_drift_idx = rows[start_idx].index
        start_mode = get_highest_mode(rows[start_drift_idx].mode)
        start_from_stability = rows[start_drift_idx].index - rows[start_drift_idx].stability
        if rows[start_from_stability].group == start_mode:
            # if start_idx == 264:
            #     print("here")
            start_drift_idx = start_from_stability
        elif rows[start_drift_idx].group != start_mode:
            # if start_idx == 264:
            #     print("here 2")
            start_drift_idx = lookback_nearest_drift_idx(rows=rows, drift_idx=start_drift_idx, key="group", criteria=start_mode)
            start_drift_idx = lookback_consecutive_drift_idx(rows=rows, drift_idx=start_drift_idx, key="group", criteria=start_mode)
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
            # print(start_idx, end_idx, "|", start_mode, end_mode)
            # print(start_idx, rows[start_idx].group)
            # if rows[start_idx].group != start_mode:
            #     ...
            # if start_idx == 235:
            #     print("here")
            # new_start_drift_idx = lookback_nearest_drift_idx(rows=rows, drift_idx=start_drift_idx, key="group", criteria=start_mode)
            new_start_drift_idx = start_drift_idx
            # new_start_drift_idx = lookback_consecutive_drift_idx(rows=rows, drift_idx=new_start_drift_idx, key="group", criteria=start_mode)
            new_end_drift_idx = end_drift_idx
            if rows[end_drift_idx].group != start_mode:
                new_end_drift_idx = lookback_nearest_drift_idx(rows=rows, drift_idx=end_drift_idx, key="group", criteria=start_mode)
            new_end_drift_idx = lookahead_consecutive_drift_idx(rows=rows, drift_idx=new_end_drift_idx, key="group", criteria=start_mode)

            intermediate_rows = rows[start_drift_idx + 1 : end_drift_idx]
            # print([c.index for c in intermediate_rows])
            # potential_c_modes_l: List[int] = [0]
            count_less_than_half = 0
            for c in intermediate_rows:
                # if c.index in caches_modes:
                #     potential_c_modes = caches_modes[c.index]
                # else:
                potential_c_modes = get_similar_modes(mode=c.mode, criteria=start_mode, diff=1, include_self=True)
                # caches_modes[c.index] = potential_c_modes
                sum_potential_c_modes = sum(potential_c_modes.values())
                if sum_potential_c_modes < (ROLLING_WINDOW / 2):
                    count_less_than_half += 1

            if count_less_than_half > 8:
                drifts.append((new_start_drift_idx, new_end_drift_idx, "recurring"))
                recurring_drifts.append((new_start_drift_idx, new_end_drift_idx))
        # else:
        elif abs(start_mode - end_mode) > 1:
            is_gradual = False
            end_current_drift_idx = start_idx
            intermediate_rows = rows[end_current_drift_idx + 1 : end_drift_idx + 1]

            ### CHECKING IF GRADUAL
            potential_c_modes_l: List[int] = [0]
            concepts: List[Tuple[int, int]] = []
            start_gradual_idx = 0

            for i, (c, n) in enumerate(zip(intermediate_rows, intermediate_rows[1:])):
                if c.index in caches_modes:
                    potential_c_modes = caches_modes[c.index]
                else:
                    potential_c_modes = get_similar_modes(mode=c.mode, criteria=end_mode, diff=1, include_self=True)
                    caches_modes[c.index] = potential_c_modes
                if n.index in caches_modes:
                    potential_n_modes = caches_modes[n.index]
                else:
                    potential_n_modes = get_similar_modes(mode=n.mode, criteria=end_mode, diff=1, include_self=True)
                    caches_modes[n.index] = potential_n_modes
                sum_potential_c_modes = sum(potential_c_modes.values())
                sum_potential_n_modes = sum(potential_n_modes.values())
                potential_c_modes_l.append(sum_potential_c_modes)

                if start_gradual_idx == 0 and sum_potential_c_modes >= (ROLLING_WINDOW / 2):
                    start_gradual_idx = c.index

                if sum_potential_c_modes >= (ROLLING_WINDOW / 2) and sum_potential_n_modes >= (ROLLING_WINDOW / 2):
                    continue

                if sum_potential_c_modes < (ROLLING_WINDOW / 2):
                    if start_gradual_idx != 0:
                        end_gradual_idx = c.index
                        concepts.append((start_gradual_idx, end_gradual_idx))
                        start_gradual_idx = 0

            if start_gradual_idx != 0:
                concepts.append((start_gradual_idx, intermediate_rows[-1].index))

            cleaned_concepts = collapse_intervals(intervals=concepts, threshold=ROLLING_WINDOW)
            is_gradual = len(cleaned_concepts) > 1

            # if start_idx == 345 and end_idx == 2607:
            #     print(start_idx, end_idx, "|", concepts, cleaned_concepts, potential_c_modes_l)
            #     print(cleaned_concepts)

            if is_gradual:
                new_start_drift_idx = lookback_consecutive_drift_idx(rows=rows, drift_idx=start_drift_idx, key="group", criteria=start_mode)
                if rows[end_drift_idx].group != end_mode:
                    new_end_drift_idx = lookback_nearest_drift_idx(rows=rows, drift_idx=end_drift_idx, key="group", criteria=end_mode)
                new_end_drift_idx = lookahead_consecutive_drift_idx(rows=rows, drift_idx=end_drift_idx, key="group", criteria=end_mode)
                drifts.append((new_start_drift_idx, new_end_drift_idx, "gradual"))
            else:

                ### CHECKING IF INCREMENTAL (INCREASING OR DECREASING)
                should_be_increasing = start_mode < end_mode
                is_incremental = True

                ranges = range(start_mode, end_mode)
                modes_l: List[Dict[int, int]] = []
                for i, c in enumerate(intermediate_rows):
                    modes = {}
                    for mode in ranges:
                        modes[mode] = c.mode.get(mode, 0)
                    modes_l.append(modes)

                    # for mode, next_mode in (modes)
                    #     if should_be_increasing:

                y = []
                x = []
                x.append(range(len(modes_l)))
                x.append([1 for _ in range(len(modes_l))])

                # for i, (modes, next_modes) in enumerate(zip(modes_l, modes_l[1:])):
                for modes in modes_l:
                    similar_m_start = get_similar_modes(mode=modes, criteria=start_mode, diff=0, include_self=True)
                    # similar_nm_end = get_similar_modes(mode=next_modes, criteria=end_mode, diff=1)
                    y.append(sum(similar_m_start.values()))
                    # x.append(sum(similar_nm_end.values()))
                    # sum_similar_m_start = sum(similar_m_start.values())
                    # sum_similar_nm_end = sum(similar_nm_end.values())

                    # print("start_mode = %s (%s), end_mode = %s (%s)" % (start_mode, sum_similar_m_start, end_mode, sum_similar_nm_end))

                    # if should_be_increasing:
                # if start_drift_idx == 250:
                #     print(y)
                x = np.matrix(x).T
                y = np.matrix(y).T
                betas = ((x.T @ x).I @ x.T) @ y
                betas = np.array(betas)
                # print(betas)
                if np.abs(betas[0][0]) < 0.01:
                    is_incremental = False
                elif betas[0][0] > 0:
                    if should_be_increasing:
                        is_incremental = False
                elif betas[0][0] < 0:
                    if not should_be_increasing:
                        is_incremental = False

                if start_drift_idx == 250:
                    print(start_drift_idx, end_drift_idx, is_incremental, betas[0][0])

                ####################

                new_start_drift_idx = lookback_consecutive_drift_idx(rows=rows, drift_idx=start_drift_idx, key="group", criteria=start_mode)
                if rows[end_drift_idx].group != end_mode:
                    new_end_drift_idx = lookback_nearest_drift_idx(rows=rows, drift_idx=end_drift_idx, key="group", criteria=end_mode)
                new_end_drift_idx = lookahead_consecutive_drift_idx(rows=rows, drift_idx=end_drift_idx, key="group", criteria=end_mode)

                diff_mode = abs(start_mode - end_mode)
                threshold = 1
                if start_mode == 0 or end_mode == 0:
                    threshold = 2
                if diff_mode > threshold:
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
                    drifts.append((new_start_drift_idx, new_end_drift_idx, "incremental"))

    for start, end, type in drifts:
        print("==== %s %s %s" % (start, end, type))


def check_is_incremental(l: List[int]) -> bool:
    if len(l) < 2:
        return False

    if l[0] == l[-1]:
        return False

    cleaned_l: List[int] = []
    for i in range(len(l)):
        if i == 0:
            cleaned_l.append(l[i])
        else:
            if l[i] != l[i - 1]:  # and abs(l[i] - l[i - 1]) > 1:
                cleaned_l.append(l[i])

    if len(cleaned_l) < 2:
        return False

    should_be_increasing = cleaned_l[0] < cleaned_l[-1]
    is_incremental = True
    roughly_similar_count = 0
    for i, (c, n) in enumerate(zip(cleaned_l, cleaned_l[1:])):
        diff = abs(c - n)
        if should_be_increasing:
            if c > n and diff > 1:
                # print("NOT INCREASING")
                is_incremental = False
                break
            if diff == 1:
                roughly_similar_count += 1
        else:
            if c < n and diff > 1:
                # print("NOT DECREASING")
                is_incremental = False
                break
            if diff == 1:
                roughly_similar_count += 1

    # if roughly_similar_count > len(cleaned_l) / 2:
    #     is_incremental = False

    return is_incremental


def check_is_incremental_v2(l: List[int]) -> bool:
    if len(l) < 2:
        return False

    if l[0] == l[-1]:
        return False

    result = mk.original_test(l)
    return (result.trend == "increasing" or result.trend == "decreasing") and result.p < 0.05


# def is_range_incremental_drift(rows: List[Row], start_idx: int, end_idx: int) -> bool:
#     l: List[int] = []
#     for idx in range(start_idx, end_idx):
#         hm = get_highest_mode(rows[idx].mode)
#         l.append(hm)

#     return check_is_incremental(l)


def process2(rows: List[Row], output_dir: Path):
    # _start_idx = 5230
    # _end_idx = 5330
    _start_idx = 4780
    _end_idx = 5390

    modes = [get_highest_mode(row.mode) for row in rows[_start_idx:_end_idx]]
    values = [row.value for row in rows[_start_idx:_end_idx]]
    results: List[Tuple[int, int]] = []

    i = 0
    # for i in range(len(modes)):
    while i < len(modes):
        # for j in range(i + 1, len(modes)):
        j = i + 1
        while j < len(modes):
            # if results_table[i][j]:
            #     continue

            if j - i > DRIFT_THRESHOLD:
                incremental = check_is_incremental(modes[i:j])
                incremental_v2 = check_is_incremental_v2(modes[i:j])
                if incremental and incremental_v2:
                    print("==== %s %s" % (i + _start_idx, j + _start_idx), flush=True)
                    results.append((i + _start_idx, j + _start_idx))
                    i = j
                    break

            j += 1
        i += 1

    # for start, end in results:
    #     print("==== %s %s" % (start, end))


def process3(rows: List[Row], output_dir: Path):
    # _start_idx = 4790
    # _end_idx = 4835
    _start_idx = 5230
    _end_idx = 5330
    modes = [get_highest_mode(row.mode) for row in rows[_start_idx:_end_idx]]
    print(modes, check_is_incremental(modes), check_is_incremental_v2(modes))


def main():
    if args.i == 1:
        data_dir = Path("/home/cc/projects/clio/data-test")
    elif args.i == 2:
        data_dir = Path("/home/cc/projects/clio/data-test2")
    else:
        data_dir = Path("/home/cc/projects/clio/data-test3")

    rmdir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # with open("/home/cc/projects/clio/runs/raw/tencent/drift/1548/1m/iops/st-15.gt-250.go-50.dt-50.rw-10/raw.dat") as f:
    rows: List[Row] = []
    data_path = "/home/cc/projects/clio/runs/raw/tencent/stats/1548/by-window/raw/real/1m/iops.dat"
    if args.i == 1:
        data_path = "/home/cc/projects/clio/raw.tsv"
    elif args.i == 2:
        data_path = "/home/cc/projects/clio/raw2.tsv"

    with open(data_path) as f:
        for i, line in enumerate(f):
            if args.i == 1 or args.i == 2:
                if i == 0:
                    continue
            data = float(line)
            g = determine_group(data, threshold=GROUP_THRESHOLD, offset=GROUP_OFFSET)
            row = Row(index=i - 1 if (args.i == 1 or args.i == 2) else i, value=data, stability=0, stability_group=0, group=g, mode={}, next_mode={})
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
                # potential_modes = [row.mode.get(row.group + diff, 0) for diff in [-1, 1, 2, -1]]
                potential_modes = get_similar_modes(mode=row.mode, criteria=row.group, diff=2)
                sum_potential_modes = sum(potential_modes.values())
                if prev_group_mode > (ROLLING_WINDOW / 2) or sum_potential_modes > (ROLLING_WINDOW / 2):
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
            f.write(f"{row.index}\t{row.value}\t{row.stability}\t{row.stability_group}\t{row.group}\t{row.prev_group}\t{row.mode}\t{row.next_mode}\n")

    process(rows, data_dir)
    # process3(rows, data_dir)
    # process2(rows, data_dir)


if __name__ == "__main__":
    main()
