#!/usr/bin/env bash

set -e

WINDOWS=("1m" "5m" "10m" "30m" "1h" "2h" "6h" "12h" "1d")
VOLUMES=(1282 1360 1488 1063 1326 1548)
METRICS=(
  # iops
  "iops"
  "read_iops"
  "write_iops"
  # iat
  "iat_avg" 
  "read_iat_avg"
  "write_iat_avg"
  # size
  "size_avg"
  "read_size_avg"
  "write_size_avg"
)

_sanity_check_() {
  assert_ret "$(command -v cmake)" "cmake not found"
  assert_ret "$(command -v ninja)" "ninja not found"  
  assert_ret "$(command -v gnuplot)" "gnuplot not found"
  assert_ret "$(command -v parallel)" "parallel not found"
  assert_ret "$(command -v gs)" "gs not found"
  pushd "$CLIO/trace-utils" >/dev/null
  check_done_ret "build" "Sanity check done" || { popd >/dev/null; return 0; }
  if [ -d build ]; then
      mkdir -p build
  fi
  pushd build >/dev/null
      cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
      ninja
  popd >/dev/null
  mark_done build
  popd >/dev/null
}

count_volume_map() {
  local input output
  input=$(parse_opt_req "input:i" "$@")
  output=$(parse_opt_default "output:o" "runs/raw/tencent/volume_count-map" "$@")
  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")
  mkdir -p "$output"

  check_done_ret "$output" "count_volume_map is done, check $output" || return 0

  pushd "$CLIO/trace-utils/build/app" >/dev/null
  ./trace-utils tencent count-volume-map --input "$input" --output "$output"
  popd >/dev/null

  mark_done "$output"
}

count_volume_reduce() {
  local input output
  input=$(parse_opt_default "input:i" "runs/raw/tencent/volume_count-map" "$@")
  output=$(parse_opt_default "output:o" "runs/raw/tencent/volume_count-reduce/summary.csv" "$@")
  input=$(realpath "$input")
  output=$(canonicalize_path "$output")
  parent_output=$(dirname "$output")
  mkdir -p "$parent_output"

  check_done_ret "$parent_output" "count_volume_reduce is done, check $output" || return 0

  pushd "$CLIO/trace-utils/build/app" >/dev/null
  ./trace-utils tencent count-volume-reduce --input "$input" --output "$output"
  popd >/dev/null

  mark_done "$parent_output"
}

pick_volume() {
  local input volume output
  input=$(parse_opt_req "input:i" "$@")
  volume=$(parse_opt_req "volume:v" "$@")
  output=$(parse_opt_default "output:o" "runs/raw/tencent/picked/$volume" "$@")
  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")
  mkdir -p "$output"

  check_done_ret "$output" "Picking device for $volume is done, check $output" || return 0

  log_info "Picking device for $volume from $input to $output"

  pushd "$CLIO/trace-utils/build/app" >/dev/null
  ./trace-utils tencent pick-volume --input "$input" --output "$output" --volume "$volume"
  popd >/dev/null

  mark_done "$output"
}

split() {
  local input output window
  input=$(parse_opt_req "input:i" "$@")
  output=$(parse_opt_req "output:o" "$@")
  window=$(parse_opt_default "window:w" "1m" "${@}")
  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")

  mkdir -p "$output"

  check_done_ret "$output" || return 0

  log_info "Splitting $input to $output with window $window"

  pushd "$CLIO/trace-utils/build/app" >/dev/null
  ./trace-utils tencent split --input "$input" --output "$output" --window "$window"
  popd >/dev/null

  mark_done "$output"
}

calc_raw_characteristic() {
  local input output window
  input=$(parse_opt_req "input:i" "$@")
  output=$(parse_opt_req "output:o" "$@")
  window=$(parse_opt_default "window:w" "1m" "$@")

  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")
  
  mkdir -p "$output"

  check_done_ret "$output" || return 0

  log_info "Calculating characteristic for $input to $output with window $window"

  pushd "$CLIO/trace-utils/build/app" >/dev/null
  ./trace-utils stats calculate raw-trace --input "$input" --output "$output" --window "$window"
  popd >/dev/null

  # mark_done "$output"
}

calc_characteristics() {
  local input output
  input=$(parse_opt_req "input:i" "$@")
  output=$(parse_opt_req "output:o" "$@")
  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")

  pushd "$CLIO/trace-utils/build/app" >/dev/null
  for window in "${WINDOWS[@]}"; do
    output_window="$output/$window"
    mkdir -p "$output_window"
    exec_report calc_characteristic --input "$input" --output "$output_window" --window "$window"
  done
  popd >/dev/null
}

plot_characteristic_cdf() {
  local input output
  input=$(parse_opt_req "input:i" "$@")
  output=$(parse_opt_req "output:o" "$@")
  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")
  mkdir -p "$output"

  log_info "Plotting CDF characteristic for $input to $output"

  # pushd "$CLIO/trace-utils/build/app" >/dev/null
  # ./target/release/plot_characteristic_cdf --input "$input" --output "$output"
  # popd >/dev/null
}

plot_characteristic_kde() {
  # _sanity_check_
  local input output
  input=$(parse_opt_req "input" "$@")
  output=$(parse_opt_req "output:o" "$@")
  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")
  mkdir -p "$output"

  log_info "Plotting KDE characteristic for $input to $output"

  # pushd "$CLIO/trace-utils" >/dev/null
  # ./target/release/plot_characteristic_kde --input "$input" --output "$output"
  # popd >/dev/null
}

generate_stats() {
  local input output
  input=$(parse_opt_req "input:i" "$@")
  output=$(parse_opt_req "output:o" "$@")
  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")
  mkdir -p "$output"

  check_done_ret "$output" || return 0

  log_info "Generating stat for $input to $output"

  # pushd "$CLIO/trace-utils" >/dev/null
  # ./target/release/generate_stats --input "$input" --output "$output"
  # popd >/dev/null

  # mark_done "$output"
}

temp_pipe() {
  # Run pick device, split and calc_characteristics in a pipeline

  local input output
  input=$(parse_opt_req "input:i" "$@")
  output=$(parse_opt_req "output:o" "$@")
  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")

  set +e

  for volume in "${VOLUMES[@]}"; do
    log_info "Processing volume $volume"
    exec_report pick_device --input "$input" --volume "$volume" --output "$output/picked/$volume"
    exec_report split --input "$output/picked/$volume" --output "$output/split/$volume"
    exec_report calc_characteristics --input "$output/split/$volume" --output "$output/characteristic/$volume"
    exec_report generate_stats --input "$output/characteristic/$volume" --output "$output/stats/$volume"
    # for window in "${WINDOWS[@]}"; do
    #   # exec_report cdf_plot --input "$output/stats/$volume/$window" --output "$output/plot-cdf/$volume/$window"
    #   for metric in "${METRICS[@]}"; do
    #     exec_report drift_finder --input "$output/stats/$volume/by-window/raw/real/$window/$metric.dat" --output "$output/drift/$volume/$window/$metric"
    #     # exit
    #   done
    #   exit
    # done
    # TODO: fix cdf_plot
    # exec_report cdf_plot --input "$output/stats/$volume" --output "$output/plot-cdf/$volume"
    # exec_report plot_characteristic_cdf --input "$output/characteristic/$volume" --output "$output/plot-cdf/$volume"
    # exec_report plot_characteristic_kde --input "$output/characteristic/$volume" --output "$output/plot-kde/$volume"
    # exit
  done

  set -e
}

drift_finder() {
  local input output
  input=$(parse_opt_req "input:i" "$@")
  output=$(parse_opt_req "output:o" "$@")
  # diff_threshold=$(parse_opt_default "diff-threshold:dt" "0.05" "$@")
  stability_threshold=$(parse_opt_default "stability-threshold:st" "14" "$@")
  group_threshold=$(parse_opt_default "group-threshold:gt" "250" "$@")
  group_offset=$(parse_opt_default "group-offset:go" "50" "$@")
  drift_threshold=$(parse_opt_default "drift-threshold:dt" "50" "$@")
  rolling_window=$(parse_opt_default "rolling-window:rw" "10" "$@")

  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")

  check_done_ret "$output" || return 0

  final_output_path="$output/st-${stability_threshold}.gt-${group_threshold}.go-${group_offset}.dt-${drift_threshold}.rw-${rolling_window}"

  log_info "Finding drift for $input to $final_output_path"
  mkdir -p "$final_output_path"

  # pushd "$CLIO/trace-utils" >/dev/null
  # ./target/release/drift_finder_v2 --input "$input" --output "$final_output_path" --stability-threshold "$stability_threshold" --group-threshold "$group_threshold" --group-offset "$group_offset" --drift-threshold "$drift_threshold" --rolling-window "$rolling_window"
  # popd >/dev/null

  # mark_done "$output"
}

_sanity_check_

# +=================+
# |    __main__     |
# +=================+

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  [[ -z "${CLIO}" ]] && echo "CLIO env is not set" && exit 1
  # shellcheck disable=SC1091
  source "${CLIO}/util.sh"
  dorun "${@@Q}"
fi
