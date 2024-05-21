#!/usr/bin/env bash

NUM_JOBS=16

# FILES=$(ls -l /mnt/dev0/Data/tencent-extract/cbs_trace1/atc_2020_trace/trace_ori/*)
# NUM_FILES=$(find /mnt/dev0/Data/tencent-extract/cbs_trace1/atc_2020_trace/trace_ori/ -type f | wc -l) # extracted file
NUM_FILES=$(find /mnt/dev0/Data/tencent/parts -type f -name "*.tgz" | wc -l) # tar file
INDEX=$(seq 0 $((NUM_FILES - 1)))
VOLUME=1282
# VOLUME=25896
OUTPUT_DIR=filtered_devices/$VOLUME
mkdir -p $OUTPUT_DIR

export VOLUME OUTPUT_DIR

# echo $FILENAMES

cargo build --release

dojob() {
    local idx=$1
    # FILES=($(ls /mnt/dev0/Data/tencent-extract/cbs_trace1/atc_2020_trace/trace_ori/* | sort -V)) # extracted file
    FILES=($(find /mnt/dev0/Data/tencent/parts -type f -name "*.tgz" | sort -V)) # tar file
    # idx=$((idx - 1))
    local file="${FILES[$idx]}"
    local filename="$OUTPUT_DIR/chunk_$idx.tar.gz"
    echo "Processing idx=$idx, file=$file to output $filename"
    ./target/release/tencent_filter_device "$file" "$filename" "$VOLUME"
}

export -f dojob

# echo $INDEX

parallel -j $NUM_JOBS --line-buffer --tag --joblog $OUTPUT_DIR/log.txt --progress \
    "dojob {}" ::: "${INDEX[@]}"

