#!/usr/bin/env bash

set -e

WINDOWS=("1m" "5m" "10m" "30m" "1h" "2h" "6h" "12h" "1d")
# VOLUMES=(1282 1360 1488 1063 1326 1548)
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

# ./r s/raw/tectonic.sh _sanity_check_
_sanity_check_() {
  assert_ret "$(command -v cmake)" "cmake not found"
  assert_ret "$(command -v ninja)" "ninja not found"  
  assert_ret "$(command -v gnuplot)" "gnuplot not found"
  assert_ret "$(command -v parallel)" "parallel not found"
  assert_ret "$(command -v gs)" "gs not found"
  pushd "$CLIO/trace-utils" >/dev/null
  check_done_ret "build" "Sanity check done" || { popd >/dev/null; return 0; }
  if [ ! -d build ]; then
      mkdir -p build
  fi
  
  pushd build >/dev/null
      cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
      ninja
  popd >/dev/null
  mark_done build
  popd >/dev/null
}

#./r s/raw/tectonic.sh split -i ./raw-data/201910/Region1/testing -o ./runs/raw/fb -m 1m
split() {
  local input output window
  input=$(parse_opt_req "input:i" "$@")
  output=$(parse_opt_req "output:o" "$@")
  window=$(parse_opt_default "window:w" "1m" "${@}")
  input=$(canonicalize_path "$input")
  output=$(canonicalize_path "$output")
  output="$output/$window"

  mkdir -p "$output"

  check_done_ret "$output" || return 0

  log_info "Splitting $input to $output with window $window"

  pushd "$CLIO/trace-utils/build/app" >/dev/null
  ./trace-utils tectonic split --input "$input" --output "$output" --window "$window"
  popd >/dev/null

  mark_done "$output"
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
