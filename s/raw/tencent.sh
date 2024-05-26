#!/usr/bin/env bash

set -e

WINDOWS=("1m" "5m" "10m" "30m" "1h" "2h" "6h" "12h" "1d")
VOLUMES=(1282 1360 1488 1063 1326 1548)

_sanity_check_() {
    assert_ret "$(command -v cargo)" "cargo not found"
    pushd "$CLIO/trace-utils" > /dev/null
    cargo build --release
    popd > /dev/null
    # assert_ret "$(command -v parallel)" "parallel not found"
}


get_device_count() {
    # _sanity_check_
    
    # local data_dir output num_jobs
    local data_dir output pattern
    data_dir=$(parse_opt_req "data:d" "$@")
    output=$(parse_opt_default "output:o" "runs/raw/tencent/volume_count" "$@")
    pattern=$(parse_opt_default "pattern:p" "*.tgz" "$@")
    # num_jobs=$(parse_opt_default "num-jobs:n" 16 "$@")
    output=$(canonicalize_path "$output")
    mkdir -p "$output"

    check_done_ret "$output" || return 0

    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/tencent_volume_count --input "$data_dir" --output "$output" --pattern "$pattern"
    popd > /dev/null

    mark_done "$output"
}

get_device_count_summary() {
    # _sanity_check_
    local data_dir output
    data_dir=$(parse_opt_default "data:d" "runs/raw/tencent/volume_count" "$@")
    output=$(parse_opt_default "output:o" "runs/raw/tencent/volume_count-summary/summary.csv" "$@")
    data_dir=$(realpath "$data_dir")
    output=$(canonicalize_path "$output")
    parent_output=$(dirname "$output")
    mkdir -p "$parent_output"

    check_done_ret "$output" || return 0

    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/tencent_device_joiner --input "$data_dir" --output "$output"
    popd > /dev/null

    mark_done "$output"
}

pick_device() {
    # _sanity_check_
    local data_dir volume output
    data_dir=$(parse_opt_req "data:d" "$@")
    volume=$(parse_opt_req "volume:v" "$@")
    output=$(parse_opt_default "output:o" "runs/raw/tencent/picked/$volume" "$@")
    data_dir=$(canonicalize_path "$data_dir")
    output=$(canonicalize_path "$output")
    mkdir -p "$output"

    check_done_ret "$output" || return 0

    echo "Picking device for $volume from $data_dir to $output"

    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/tencent_pick_device --input "$data_dir" --output "$output" --volume "$volume"
    popd > /dev/null

    mark_done "$output"
}

split() {
    # _sanity_check_
    local data_dir output window
    data_dir=$(parse_opt_req "data:d" "$@")
    output=$(parse_opt_req "output:o" "$@")
    window=$(parse_opt_default "window:w" "1m" "$@")
    data_dir=$(canonicalize_path "$data_dir")
    output=$(canonicalize_path "$output")
    mkdir -p "$output"

    check_done_ret "$output" || return 0

    echo "Splitting $data_dir to $output with window $window"
    
    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/tencent_split_window --input "$data_dir" --output "$output" --window "$window"
    popd > /dev/null

    mark_done "$output"
}

calc_characteristic() {
    # _sanity_check_
    local data_dir output window
    data_dir=$(parse_opt_req "data:d" "$@")
    output=$(parse_opt_req "output:o" "$@")
    window=$(parse_opt_default "window:w" "1m" "$@")

    data_dir=$(canonicalize_path "$data_dir")
    output=$(canonicalize_path "$output")
    mkdir -p "$output"

    check_done_ret "$output" || return 0

    echo "Calculating characteristic for $data_dir to $output"

    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/calc_characteristic --input "$data_dir" --output "$output" --window "$window"
    popd > /dev/null

    mark_done "$output"
}

calc_characteristics() {
    # _sanity_check_
    local data_dir output
    data_dir=$(parse_opt_req "data:d" "$@")
    output=$(parse_opt_req "output:o" "$@")
    data_dir=$(canonicalize_path "$data_dir")
    output=$(canonicalize_path "$output")

    # echo "Calculating characteristics for $data_dir to $output"
    pushd "$CLIO/trace-utils" > /dev/null
    for window in "${WINDOWS[@]}"; do
        output_window="$output/$window"
        mkdir -p "$output_window"
        exec_report calc_characteristic --data "$data_dir" --output "$output_window" --window "$window"
    done
    popd > /dev/null
}

plot_characteristic_cdf() {
    # _sanity_check_
    local data_dir output
    data_dir=$(parse_opt_req "data:d" "$@")
    output=$(parse_opt_req "output:o" "$@")
    data_dir=$(canonicalize_path "$data_dir")
    output=$(canonicalize_path "$output")
    mkdir -p "$output"

    echo "Plotting CDF characteristic for $data_dir to $output"

    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/plot_characteristic_cdf --input "$data_dir" --output "$output"
    popd > /dev/null
}

plot_characteristic_kde() {
    # _sanity_check_
    local data_dir output
    data_dir=$(parse_opt_req "data:d" "$@")
    output=$(parse_opt_req "output:o" "$@")
    data_dir=$(canonicalize_path "$data_dir")
    output=$(canonicalize_path "$output")
    mkdir -p "$output"

    echo "Plotting KDE characteristic for $data_dir to $output"

    pushd "$CLIO/trace-utils" > /dev/null
    ./target/release/plot_characteristic_kde --input "$data_dir" --output "$output"
    popd > /dev/null
}

temp_pipe() {
    # Run pick device, split and calc_characteristics in a pipeline

    local data_dir output
    data_dir=$(parse_opt_req "data:d" "$@")
    output=$(parse_opt_req "output:o" "$@")
    data_dir=$(canonicalize_path "$data_dir")
    output=$(canonicalize_path "$output")

    set +e

    for volume in "${VOLUMES[@]}"; do
        echo "Processing volume $volume"
        exec_report pick_device --data "$data_dir" --volume "$volume" --output "$output/picked/$volume"
        exec_report split --data "$output/picked/$volume" --output "$output/split/$volume"
        exec_report calc_characteristics --data "$output/split/$volume" --output "$output/characteristic/$volume"
        exec_report plot_characteristic_cdf --data "$output/characteristic/$volume" --output "$output/plot-cdf/$volume"
        exec_report plot_characteristic_kde --data "$output/characteristic/$volume" --output "$output/plot-kde/$volume"
        exit
    done

    set -e
}

_sanity_check_

# +=================+
# |    __main__     |
# +=================+

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    [[ -z "${CLIO}" ]] && echo "CLIO env is not set" && exit 1
    # shellcheck disable=SC1091
    source "${CLIO}/util.sh"
    dorun "$@"
fi