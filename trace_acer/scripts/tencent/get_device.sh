#!/usr/bin/env bash

OUTPUT_DIR=devicesbb
mkdir -p $OUTPUT_DIR
NUM_JOBS=16

# export FILES=$(ls /mnt/dev0/Data/tencent-extract/cbs_trace1/atc_2020_trace/trace_ori/*) # extracted file
export FILES=$(ls /mnt/dev0/Data/tencent/parts/*.tgz) # tar file
export INDEX=$(seq 0 $(echo $FILES | wc -w))
export FILENAMES=($(echo $INDEX | tr ' ' '\n' | awk '{print "'$OUTPUT_DIR'/device_"$1".csv"}'))

# echo $FILES
# echo $INDEX

# echo $FILENAMES

cargo build --release

dojob() {
    local file=$1
    local filename=$2
    echo "Processing $file to output $filename"
    # cargo run --release $file $filename
    ./target/release/tencent_find_device "$file" "$filename"
}

export -f dojob

parallel -j $NUM_JOBS --line-buffer --tag --joblog $OUTPUT_DIR/log.txt --progress \
    "dojob {} {}" ::: "${FILES[@]}" :::+ "${FILENAMES[@]}"

