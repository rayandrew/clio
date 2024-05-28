#!/usr/bin/env bash

set -e

export PATH="${PATH}:${CLIO}/bin:${CLIO}/utils"
export GNUPLOT_LIB="${GNUPLOT_LIB}:${CLIO}/utils"

# shellcheck source=util.sh
source util.sh

py() {
  python
}

py_ver() {
  python --version
}

py_loc() {
  which python
}

check_cuda() {
  if ! command -v nvcc &>/dev/null; then
    _log_err "CUDA is not installed"
    return 1
  fi
  echo "CUDA version: $(nvcc --version)"
  python -c "import torch; print(torch.cuda.is_available())"
}

test() {
  echo "$@"
}

test2() {
  local dataset model output_path
  dataset=$(parse_opt "dataset:d" "$@")
  model=$(parse_opt "model:m" "$@")
  output_path=$(parse_opt "output-path:o" "$@")
  echo "dataset: $dataset"
  echo "model: $model"
  echo "output_path: $output_path"
}

test3() {
  local arg
  parse_arg arg "$@"
  for i in "${!arg[@]}"; do
    echo "$i: ${arg[$i]}"
  done
}

test4() {
  local dataset
  dataset=$(parse_opt_req "dataset:d" "$@")
  echo "dataset: $dataset"
}

test5() {
  local force
  force=$(parse_opt_flag "force:f" "$@")
  echo "Force: $force, truthy = $(is_truthy "$force"), falsy = $(is_falsy "$force")"
}


cdf_plot() {
  # _sanity_check_
  local data_dir output
  data_dir=$(parse_opt_req "data:d" "$@")
  output=$(parse_opt_req "output:o" "$@")
  pattern=$(parse_opt_default "pattern:p" "" "$@")
  data_dir=$(canonicalize_path "$data_dir")
  output=$(canonicalize_path "$output")
  parent_output=$(dirname "$output")
  mkdir -p "$parent_output"

  pushd "$CLIO/trace-utils" >/dev/null
  if [[ -z "$pattern" ]]; then
    log_info "Plotting CDF for $data_dir to $output"
    gnuplot -c plot/cdf.plot "$data_dir" "$output"
  else
    log_info "Plotting CDF for $data_dir to $output with pattern $pattern"
    gnuplot -c plot/cdf.plot "$data_dir" "$output" "$pattern"
  fi
  # change the output extension to png
  png_output="${output%.*}.png"
  gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=png16m -r1000 -sOutputFile="$png_output" "$output"
  # pdf_output="${output%.*}.pdf"
  # gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=pdfwrite -sOutputFile="$pdf_output" "$output"
  popd >/dev/null
}

line_plot() {
  # _sanity_check_
  local data_dir output
  data_dir=$(parse_opt_req "data:d" "$@")
  output=$(parse_opt_req "output:o" "$@")
  pattern=$(parse_opt_default "pattern:p" "" "$@")
  y_label=$(parse_opt_default "y-label:y" "" "$@")
  data_dir=$(canonicalize_path "$data_dir")
  output=$(canonicalize_path "$output")

  if [[ -z "$pattern" ]]; then
    basepath=$(basename "$data_dir")
    basepath="${basepath%.*}"
    output_path="$output/$basepath.eps"
    parent_output=$(dirname "$output_path")
    mkdir -p "$parent_output"
    log_info "Plotting CDF for $data_dir to $output_path with y_label=$y_label"
    pushd "$CLIO/trace-utils" >/dev/null
    gnuplot -c plot/line.plot "$data_dir" "$output_path" "$y_label"
    popd >/dev/null
    png_output="${output_path%.*}.png"
    gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=png16m -r1000 -sOutputFile="$png_output" "$output_path"
  else
    log_info "Plotting CDF for $data_dir to $output with pattern $pattern"
    for f in $(find "$data_dir" -type f -name "$pattern"); do
      log_info "Processing $f"
      line_plot --data "$f" --output "$output" --y-label "$y_label"
    done
  fi
}

# +=================+
# |    __main__     |
# +=================+

dorun "$@"

