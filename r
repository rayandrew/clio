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

  # pushd "$CLIO/trace-utils" >/dev/null
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
  # popd >/dev/null
}

cdf_from_replay_data() {
  local data output min max precision
  data=$(parse_opt_req "data:d" "$@")
  output=$(parse_opt_req "output:o" "$@")
  min=$(parse_opt_default "min:m" "0" "$@")
  max=$(parse_opt_default "max:M" "1" "$@")
  precision=$(parse_opt_default "precision:p" "0.0001" "$@")
  data=$(canonicalize_path "$data")
  output=$(canonicalize_path "$output")
  parent_output=$(dirname "$output")
  mkdir -p "$parent_output"
  rm -f "$output"

  log_info "Extracting CDF data from replay directory $data to $output"
  tmp_file=$(mktemp)
  awk -F, '{print $2}' "$data" | sort -n >"$tmp_file"
  dat_output="${output%.*}.dat"
  rm -f "$dat_output"
  awk -v min="$min" -v max="$max" -v precision="$precision" -v output="$dat_output" -f "${CLIO}/utils/cdf.awk" "$tmp_file"
  eps_output="${output%.*}.eps"
  cdf_plot -d "$dat_output" -o "$eps_output"
  rm "$tmp_file"
}

cdf_from_replay_dir() {
  local data output min max precision
  data=$(parse_opt_req "data:d" "$@")
  output=$(parse_opt_req "output:o" "$@")
  min=$(parse_opt_default "min:m" "0" "$@")
  max=$(parse_opt_default "max:M" "1" "$@")
  precision=$(parse_opt_default "precision:p" "0.0001" "$@")
  data=$(canonicalize_path "$data")
  output=$(canonicalize_path "$output")
  parent_output=$(dirname "$output")
  mkdir -p "$parent_output"
  # rm -f "$output"

  log_info "Extracting CDF data from replay directory $data to $output"
  for f in $data/*; do
    if [[ "$f" == *".stats" ]]; then
      continue
    fi
    log_info "Processing $f"
    base_f=$(basename "$f")
    cdf_from_replay_data -d "$f" -o "$output/$base_f" -m "$min" -M "$max" -p "$precision"
  done
  eps_output="$output/cdf.eps"
  cdf_plot -d "$output" -o "$eps_output" -p "*.dat"
}

cdf_concat_from_replay_dir_single() {
  local data_dir output min max precision
  data_dir=$(parse_opt_req "data-dir:d" "$@")
  output=$(parse_opt_req "output:o" "$@")
  title=$(parse_opt_req "title:t" "$@")
  min=$(parse_opt_default "min:m" "0" "$@")
  max=$(parse_opt_default "max:M" "1" "$@")
  precision=$(parse_opt_default "precision:p" "0.0001" "$@")
  data_dir=$(canonicalize_path "$data_dir")
  output=$(canonicalize_path "$output")
  mkdir -p "$output"

  #find done file paths, recursive, not just in this dir
  done_file_paths=$(find "$data_dir" -type f -name "done")
  echo "$done_file_paths"

  for done_file_path in $done_file_paths; do
    # log_info "Extracting CDF data from replay directory $data_dir to $output"
    tmp_file=$(mktemp)
    # get parent of done dir
    parent_dir=$(dirname "$done_file_path")

    find "$data_dir" -type f -name "*.csv" | while read -r file; do
      if [[ -f "$file" ]]; then
        log_info "Processing file: $file"
        awk -F, '{print $2}' "$file" >> "$tmp_file"
      fi
    done
  done

  # Sort the collected data
  sort -n "$tmp_file" -o "$tmp_file"
  
  dat_output="$output/${title}.dat"
  rm -f "$dat_output"
  awk -v min="$min" -v max="$max" -v precision="$precision" -v output="$dat_output" -f "${CLIO}/utils/cdf.awk" "$tmp_file"
  eps_output="$output/${title}.eps"
  cdf_plot -d "$dat_output" -o "$eps_output"
  rm "$tmp_file"
}

cdf_concat_from_replay_dir_glob() {
  local data_dir output min max precision
  data_dir=$(parse_opt_req "data-dir:d" "$@")
  output=$(parse_opt_req "output:o" "$@")
  is_femu=$(parse_opt_flag "femu:f" "$@")
  min=$(parse_opt_default "min:m" "0" "$@")
  max=$(parse_opt_default "max:M" "1" "$@")
  precision=$(parse_opt_default "precision:p" "0.0001" "$@")
  data_dir=$(canonicalize_path "$data_dir")
  output=$(canonicalize_path "$output")
  mkdir -p "$output"

  #find done file paths, recursive, not just in this dir
  done_file_paths=$(find "$data_dir" -type f -name "done")
  echo "$done_file_paths"

  for done_file_path in $done_file_paths; do
    tmp_file=$(mktemp)
    parent_dir=$(dirname "$done_file_path")

    title_type=$(echo "$parent_dir" | grep -oP '(?<=/)(0\.5|1\.5|sudden)' || true)
    if [[ -z "$title_type" ]]; then
      echo "Title type not found"
      continue
    fi
    title_start_end=$(echo "$parent_dir" | grep -oP '\d+_\d+')

    title="$title_type-$title_start_end"

    # if flag is set, else
    echo "$is_femu"
    if [[ "$is_femu" == "1" ]]; then
      title="femu-$title"
    else
      title="realSSD-$title"
    fi


    log_info "Extracting CDF data from replay directory $parent_dir to $output. Title: $title"

    ./r cdf_concat_from_replay_dir_single -d "$parent_dir" -o "$output/$title_type/$title_start_end" -t $title --min $min --max $max
  done
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
    gnuplot -c plot/line.plot "$data_dir" "$output_path" "$y_label"
    png_output="${output_path%.*}.png"
    gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=png16m -r1000 -sOutputFile="$png_output" "$output_path"
  else
    log_info "Plotting CDF for $data_dir to $output with pattern $pattern"
    for f in $(find "$data_dir" -type f -name "$pattern"); do
      # log_info "Processing $f"
      line_plot --data "$f" --output "$output" --y-label "$y_label"
    done
  fi
}


tmp_split_plot() {
  local start end i
  start=$(parse_opt_req "start:s" "$@")
  end=$(parse_opt_req "end:e" "$@")
  i=$(parse_opt_req "index:i" "$@")

  echo "$i"

  log_info "Running split plot for $start to $end"
  if [[ "$i" == "1" ]]; then
    python sp.py -s "$start" -e "$end" -o ./dat -i 1
    ./r line_plot -d "dat/${start}_${end}.dat" -o ./tmp --y-label "IOPS"
  else
    python sp.py -s "$start" -e "$end" -o ./dat2 -i 2
    ./r line_plot -d "dat2/${start}_${end}.dat" -o ./tmp2 --y-label "IOPS"
  fi
  # python sp.py -s "$start" -e "$end" -o ./dat
  # ./r line_plot -d "dat/${start}_${end}.dat" -o ./tmp --y-label "IOPS"

  #   python sp.py -s "$start" -e "$end" -o ./dat2
  # ./r line_plot -d "dat2/${start}_${end}.dat" -o ./tmp2 --y-label "IOPS"
}

# +=================+
# |    __main__     |
# +=================+

dorun "${@@Q}"

