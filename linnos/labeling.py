#!/usr/bin/env python3

import pandas as pd
from argparse import ArgumentParser
import os
import sys
from pathlib import Path

def count_slope(y1, y2):
    return y2 - y1
# x axis = timestamp --> not used
# y axis = latency

def count_ratio(y1, y2):
    try:
        return y2 / y1
    except ZeroDivisionError:
        return 0

def getLatencyGCaffected(latency, choice):
    GC_aff = []
    if choice == "slope":
        GC_aff.append(latency[0])
    elif choice == "ratio":
        GC_aff.append(0)
    for i in range(1, len(latency)):
        if choice == "slope":
            GC_aff.append(count_slope(latency[i-1], latency[i]))
        elif choice == "ratio":
            GC_aff.append(count_ratio(latency[i-1], latency[i]))
    return GC_aff

def getTroughputGCaffected(throughput, choice):
    GC_aff = []
    if choice == "slope":
        GC_aff.append(throughput[0])
    elif choice == "ratio":
        GC_aff.append(0)
    for i in range(1, len(throughput)):
        if choice == "slope":
            GC_aff.append(count_slope(throughput[i-1], throughput[i]))
        elif choice == "ratio":
            GC_aff.append(count_ratio(throughput[i-1], throughput[i]))
    return GC_aff

def count_iqr(data):
    iqr1 = data.quantile(0.75) - data.quantile(0.25)
    iqr1_1_5 = 1.5*iqr1
    iqr1_3 = 3*iqr1
    lower_inner_fence_1 = data.quantile(0.25) - iqr1_1_5
    lower_outer_fence_1 = data.quantile(0.25) - iqr1_3
    upper_inner_fence_1 = data.quantile(0.75) + iqr1_1_5
    upper_outer_fence_1 = data.quantile(0.75) + iqr1_3
    print("Fences: ", lower_outer_fence_1, lower_inner_fence_1, upper_inner_fence_1, upper_outer_fence_1)
    return lower_outer_fence_1, lower_inner_fence_1, upper_inner_fence_1, upper_outer_fence_1

def label_lathpt_iqr(df, lat_mark_start, lat_mark_end, thpt_mark_start, thpt_mark_end, thpt_mark_middle, thpt_threshold):
    label = []
    start = False
    beforeWasWrite = False
    beforeSlope = 0
    for i in range(len(df)):
        if not start:
            # Check if this IO is write, after this could be GC affected
            if df['io_type'][i] == 0:
                beforeWasWrite = True
            # If the IO is read and it's not 4096, it might be the start, check the slope
            if df['io_type'][i] == 1 and df['size'][i] > 4096 and df['Slope_l'][i] > lat_mark_start and df['Slope_t'][i] < thpt_mark_start:
                # if beforeWasWrite == True:
                #     beforeWasWrite = False
                start = True # Mark the start
                beforeSlope = df['Slope_t'][i]
                label.append(1)
            # If the IO is read and it's not 4096 and the previous IO was write, it could be GC affected
            # If the prev IO is write but the current one having high troughput, it doesn't affected by GC
            elif beforeWasWrite == True and df['io_type'][i] == 1 and df['size'][i] > 4096 and not df['Slope_t'][i] > thpt_mark_middle:
                beforeWasWrite = False
                start = True
                beforeSlope = df['Slope_t'][i]
                label.append(1)
            # Exclude small size IO so it won't be rejected
            # Exclude write IO so it won't be rejected
            # Stay low slope
            else:
                label.append(0)
        else:
            if df['io_type'][i] == 1 and df['size'][i] > 4096:
                # If the slope goes down, it's the end of GC affected
                if df['Slope_l'][i] < lat_mark_end and df['Slope_t'][i] > thpt_mark_end:
                    if beforeWasWrite == True:
                        beforeWasWrite = False
                    start = False
                    label.append(0)
                # If the slope stays the same after started, it's still GC affected
                elif df['Slope_t'][i] >= beforeSlope-(thpt_threshold*beforeSlope) and df['Slope_t'][i] <= beforeSlope+(thpt_threshold*beforeSlope):
                    if beforeWasWrite == True:
                        beforeWasWrite = False
                    beforeSlope = df['Slope_t'][i]
                    label.append(1)
                # Not affected anymore since the difference is bigger than threshold
                else:
                    label.append(1)
            # Exclude small size IO so it won't be rejected
            # Exclude write IO so it won't be rejected
            else:
                label.append(0)
    return label

def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

# save to a file
def write_to_file(df, filePath, has_header=True):
    # The raw (replayed) traces don't have the header, so when we
    # write the filtered version, it must be consistent with the original
    df.to_csv(filePath, index=False, header=has_header, sep=',')
    print("===== output file : " + filePath)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-file", help="File path of the replayed trace",type=str)
    parser.add_argument("-output_dir", help="Folder path of the replayed trace",type=str)
    args = parser.parse_args()
    if (not args.file or not args.output_dir):
        print("    ERROR: You must provide these arguments: -file <the input trace>, folder <the folder name>")
        exit(-1)

    trace_replayed = args.file

    cols = ["ts_record","latency","io_type","size","offset","ts_submit","size_after_replay"]
    df = pd.read_csv(trace_replayed, header=None, sep=',', names=cols)
    df['throughput'] = df['size'] / df['latency']
    df = df.sort_values('ts_submit')
    df = df.reset_index(drop=True)

    df['Slope_l'] = pd.Series(getLatencyGCaffected(df['latency'], "slope"))
    df['Slope_t'] = pd.Series(getTroughputGCaffected(df['throughput'], "slope"))

    lower_outer_fence_1, lower_inner_fence_1, upper_inner_fence_1, upper_outer_fence_1 = count_iqr(df.copy(deep=True)['Slope_l'])
    df['reject'] = pd.Series(label_lathpt_iqr(df.copy(deep=True), upper_outer_fence_1, lower_outer_fence_1, df['Slope_t'].quantile(0.25), df['Slope_t'].quantile(0.75), df['Slope_t'].quantile(0.5), 0.1))

    df = df.drop(columns=['Slope_l', 'Slope_t', 'throughput'], axis=1)
    profile_name = os.path.basename(args.file)
    # parent_dir_name = os.path.basename(Path(args.file).parent)

    # profile_name = str(Path(profile_name).with_suffix('') ) # remove .trace extension
    output_dir = args.output_dir
    create_output_dir(output_dir)
    outfile_path = os.path.join(output_dir, f"{profile_name}.labeled")
    write_to_file(df, outfile_path, True)

    # df.to_csv('.'.join(trace_replayed.split('.')[:-1]) + ".marked0", index=False)
    # print("===== output file : " + '.'.join(trace_replayed.split('.')[:-1]) + ".marked0")
