#!/usr/bin/env bash

set -e

_sanity_check_() {
    assert_ret "$(command -v cargo)" "cargo not found"
    pushd "$CLIO/trace-utils" > /dev/null
    cargo build --release
    popd > /dev/null
    # assert_ret "$(command -v parallel)" "parallel not found"
}

#################
# OLD CODE
#################
# _get_device_job() {
#     local file=$1
#     local filename=$2
#     echo "Processing $file to output $filename"
#     # cargo run --release $file $filename
#     ./target/release/tencent_find_device --input "$file" --output "$filename"
# }
#################


get_device_count() {
    _sanity_check_
    
    # local data_dir output num_jobs 
    local data_dir output pattern
    data_dir=$(parse_opt_req "data:d" "$@")
    output=$(parse_opt_default "output:o" "runs/raw/tencent/volume_count" "$@")
    pattern=$(parse_opt_default "pattern:p" "*.tgz" "$@")
    # num_jobs=$(parse_opt_default "num-jobs:n" 16 "$@")
    output=$(canonicalize_path "$output")
    mkdir -p "$output"

    #################
    # OLD CODE
    #################
    # shopt -s nullglob # prevent globbing to return itself if no files found
    # shopt -s globstar # enable ** to match all files and zero or more directories and subdirectories
    # files=("$data_dir"/*.{tgz,tar.gz})
    # len=${#files[@]}
    # mapfile -t index < <(seq 0 $((len - 1)))
    # filenames=( "${index[@]/#/$output/device_}" )
    # echo "Found $len files in $data_dir"
    # echo "Filenames: ${filenames[@]}"
    # disable glob
    # shopt -u nullglob
    # shopt -u globstar
    # export -f _get_device_job
    # parallel -j "$num_jobs" --line-buffer --tag --joblog "$output/log.txt" --progress \
    #     "_get_device_job {} {}" ::: "${files[@]}" :::+ "${filenames[@]}"
    #################

    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/tencent_volume_count --input "$data_dir" --output "$output" --pattern "$pattern"
    popd > /dev/null
}

get_device_count_summary() {
    _sanity_check_
    local data_dir output
    data_dir=$(parse_opt_default "data:d" "runs/raw/tencent/volume_count" "$@")
    output=$(parse_opt_default "output:o" "runs/raw/tencent/volume_count-summary/summary.csv" "$@")
    data_dir=$(realpath "$data_dir")
    output=$(canonicalize_path "$output")
    parent_output=$(dirname "$output")
    mkdir -p "$parent_output"

    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/tencent_device_joiner --input "$data_dir" --output "$output"
    popd > /dev/null
}

pick_device() {
    _sanity_check_
    local data_dir volume output
    data_dir=$(parse_opt_req "data:d" "$@")
    volume=$(parse_opt_req "volume:v" "$@")
    output=$(parse_opt_default "output:o" "runs/raw/tencent/picked/$volume" "$@")
    data_dir=$(canonicalize_path "$data_dir")
    output=$(canonicalize_path "$output")
    mkdir -p "$output"

    echo "Picking device for $volume from $data_dir to $output"

    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/tencent_pick_device --input "$data_dir" --output "$output" --volume "$volume"
    popd > /dev/null
}

split() {
    _sanity_check_
    local data_dir output window
    data_dir=$(parse_opt_req "data:d" "$@")
    output=$(parse_opt_req "output:o" "$@")
    window=$(parse_opt_default "window:w" "1m" "$@")
    data_dir=$(canonicalize_path "$data_dir")
    output=$(canonicalize_path "$output")
    mkdir -p "$output"

    echo "Splitting $data_dir to $output with window $window"
    
    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/tencent_split_window --input "$data_dir" --output "$output" --window "$window"
    popd > /dev/null
}

calc_characteristic() {
    _sanity_check_
    local data_dir output
    data_dir=$(parse_opt_req "data:d" "$@")
    output=$(parse_opt_req "output:o" "$@")
    data_dir=$(canonicalize_path "$data_dir")
    output=$(canonicalize_path "$output")
    mkdir -p "$output"

    echo "Calculating characteristic for $data_dir to $output"

    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/calc_characteristic --input "$data_dir" --output "$output"
    popd > /dev/null

}

# +=================+
# |    __main__     |
# +=================+

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    [[ -z "${CLIO}" ]] && echo "CLIO env is not set" && exit 1
    # shellcheck disable=SC1091
    source "${CLIO}/util.sh"
    dorun "$@"
fi