#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import argparse
import csv

LEN_HIS_QUEUE = 4

# From the paper and the readme of LinnOS
# So isn't made as a required argument for running the script
LEN_PENDING_DIGIT = 3
LEN_LATENCY_DIGIT = 4

# LinnOS' feature engineering automatically only takes read I/Os and discards write I/Os
IO_READ = '1'
IO_WRITE = '0'
LABEL_ISSUE = 'i'
LABEL_COMPLETION = 'c'

def generate_raw_vec(input_path, output_path):

    with open(input_path, 'r') as input_file:

        input_csv = csv.reader(input_file)

        trace_list = []
        transaction_list = []
        index = 0
        for idx, row in enumerate(input_csv):
            if idx == 0:
                continue
            latency = int(row[1])
            type_op = row[2]
            size_ori = int(row[3])
            size = int((int(row[3])/512 + 7)/8)
            issue_ts = int(float(row[5])*1000)
            complete_ts = issue_ts+latency

            trace_list.append([size, type_op, latency, 0, index])
            transaction_list.append([index, issue_ts, LABEL_ISSUE])
            transaction_list.append([index, complete_ts, LABEL_COMPLETION])

            index += 1

    transaction_list = sorted(transaction_list, key=lambda x: x[1])

    print('trace loading completed:', len(trace_list), 'samples')

    with open(output_path, 'w') as output_file:

        count = 0
        skip = 0
        pending_io = 0
        history_queue = [[0, 0]]*LEN_HIS_QUEUE
        raw_vec = [0]*(LEN_HIS_QUEUE*2+1+1)
        # print(history_queue)
        for trans in transaction_list:

            io = trace_list[trans[0]]
            if trans[2] == LABEL_ISSUE:
                pending_io += io[0]
                io[3] = pending_io

                if io[1] == IO_READ and skip >= LEN_HIS_QUEUE:
                    count += 1
                    raw_vec[LEN_HIS_QUEUE] = io[3]
                    raw_vec[-1] = io[2]
                    for i in range(LEN_HIS_QUEUE):
                        raw_vec[i] = history_queue[i][1]
                        raw_vec[i+LEN_HIS_QUEUE+1] = history_queue[i][0]
                    output_file.write(','.join(str(x) for x in raw_vec)+'\n')

            elif trans[2] == LABEL_COMPLETION:
                pending_io -= io[0]

                if io[1] == IO_READ:
                    history_queue.append([io[2], io[3]])
                    del history_queue[0]
                    skip += 1

        # print(history_queue)
        # print(pending_io)
        print('Done:', str(count), 'vectors')

def create_output_dir(output_path):
    os.makedirs(output_path, exist_ok=True)
    return output_path

def generate_ml_vec(len_pending, len_latency, input_path, output_path):

    max_pending = (10**len_pending)-1
    max_latency = (10**len_latency)-1
    # print(max_pending, max_latency)

    with open(input_path, 'r') as input_file, open(output_path, 'w+') as output_file:

        input_csv = csv.reader(input_file)

        for rvec in input_csv:

            tmp_vec = []
            for i in range(LEN_HIS_QUEUE+1):
                pending_io = int(rvec[i])
                if pending_io > max_pending:
                    pending_io = max_pending
                tmp_vec.append(','.join(x for x in str(pending_io).rjust(len_pending, '0')))
            for i in range(LEN_HIS_QUEUE):
                latency = int(rvec[i+LEN_HIS_QUEUE+1])
                if latency > max_latency:
                    latency = max_latency
                tmp_vec.append(','.join(x for x in str(latency).rjust(len_latency, '0')))
            tmp_vec.append(rvec[-1])
            output_file.write(','.join(x for x in tmp_vec)+'\n')

def start_processing(input_path, output_real):
    len_pending = LEN_PENDING_DIGIT
    len_latency = LEN_LATENCY_DIGIT
    trace_path = input_path
    raw_path = input_path.rsplit('.', 1)[0]+'.tmp'
    
    ml_path = output_real
    # mkdir parent
    create_output_dir(os.path.dirname(ml_path))
    
    # Processing
    generate_raw_vec(trace_path, raw_path)
    generate_ml_vec(len_pending, len_latency, raw_path, ml_path)
    
    # Remove the temporary file
    os.remove(raw_path)
    
    print("===== output file : " + ml_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir_input", help="File path of the labeled trace profiles",type=str)
    parser.add_argument("-output", help="Output directory",type=str)
    args = parser.parse_args()
    if (not args.dir_input and not args.output):
        print("    ERROR: You must provide these arguments ")
        exit(-1)
        
    # find all .csv.labeled in dir_input, glob it  
    trace_profiles = []
    import glob
    # recursively in dir input, all files with .labeled
    dir_input = args.dir_input
    dir_input = os.path.dirname(dir_input)
    
    for file in glob.glob(dir_input + "/*.labeled", recursive=True):
        trace_profiles.append(file)
        
    print("\nDir input " + args.dir_input +"\nFound " + str(len(trace_profiles)))
    
    for profile_path in trace_profiles:
        basename = os.path.basename(profile_path)
        output_real = os.path.join(args.output, basename)
        print("\nProcessing " + str(profile_path))
        start_processing(profile_path, output_real)

# Example how to run:
# python feat_linnOS.py -file /mnt/extra/flashnet/data/trace_profile/nvme1n1/alibaba.cut.per_10k.least_size.686.trace