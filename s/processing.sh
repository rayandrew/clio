
#!/usr/bin/env bash

# ./r s/processing.sh compile_and_get_drifts -o ./output -i ./runs/raw/tencent/characteristic/1063/1m/characteristic.csv -m iops
compile_and_get_drifts() {
    local input_dir output_dir metric
    input_dir=$(parse_opt_req "input:i" "$@")
    output_dir=$(parse_opt_req "output:o" "$@")
    metric=$(parse_opt_req "metric:m" "$@")

    g++ -O3 -std=c++17 ./concept_finder.cpp -o ./bin/finder.out

    ./bin/finder.out $output_dir $input_dir $metric
}

# ./r s/processing.sh plot_drifts -p ./output/iops/data/ -o ./output/iops/plot/
plot_drifts() {
    local plot_data_dir=$(parse_opt_req "plot-data-dir:p" "$@")
    local output_dir=$(parse_opt_req "output:o" "$@")

    export output_dir  # Export the variable to be used by GNU parallel
    find $plot_data_dir -name "*.tsv" | parallel '
        folder_name=$(basename $(dirname {}));
        ./r line_plot --data {} --output $output_dir/$folder_name --y-label "$folder_name-$(basename {})"
    '
}

# Input Directory, /output/device/METRIC
# Output Directory, /output/device/METRIC
experiment_loop() {
  local input_dir output_dir
  input_dir=$(parse_opt_req "input:i" "$@")
  output_dir=$(parse_opt_req "output:o" "$@")

  # Loop for folders ending with /raw
  for folder in $(find $input_dir -type d -name "raw"); do
    # if no done file, skip
    if [[ ! -f $folder/done ]]; then
      echo "Not finished replaying, skipping"
      continue
    fi
    drift_range=$(basename $(dirname $folder))
    drift_type=$(basename $(dirname $(dirname $folder)))
    echo "Labeling and feature engineering: $drift_type $drift_range"

    output_label_feature_dir=$output_dir/processed/$drift_type/$drift_range

    ./r s/processing postprocess -o $output_label_feature_dir -i $folder -m iops

    echo "Training Initial: $drift_type $drift_range"
    ./r s/train initial_only --data $output_label_feature_dir -o $output_dir/experiments/$drift_type/$drift_range

    echo "Training always retrain: $drift_type $drift_range"
    ./r s/train always_retrain --data $output_label_feature_dir -o $output_dir/experiments/$drift_type/$drift_range
  done
}

# labeling + feature engineering
postprocess() {
    local input_file output_path
    input_dir=$(parse_opt_req "input:i" "$@")
    output_dir=$(parse_opt_req "output:o" "$@")

    if [[ -f $output_dir/done ]]; then
        echo "Already processed, skipping"
        return
    fi

    python -m clio.flashnet.cli.characteristic generate_v2 \
        $input_dir \
        --output $output_dir \
        --relabel

    touch $output_dir/done
}

replay_list_real_ssd() {
  local range_list data_dir output_dir
  range_list=$(parse_opt_req "range-list:r" "$@")
  data_dir=$(parse_opt_req "data-dir:d" "$@")
  output_dir=$(parse_opt_req "output:o" "$@")

  start_arr=()
  end_arr=()
  type_arr=()
  should_replay_arr=()

  # Loop through the range list (CSV file) and append to the arrays
  while IFS=, read -r start end type should_replay; do
    if [[ "$should_replay" != "y" ]]; then
      continue
    fi
    start_arr+=("$start")
    end_arr+=("$end")
    type_arr+=("$type")
    should_replay_arr+=("$should_replay")
  done < "$range_list"

  # Loop through the arrays and replay
  for i in "${!start_arr[@]}"; do
    start=${start_arr[$i]}
    end=${end_arr[$i]}
    type=${type_arr[$i]}

    echo "Replaying: \n Start: $start, End: $end, Type: $type"
    output_folder="${output_dir}/${type}/${start}_${end}/raw/"
    if [[ -f $output_folder/done ]]; then
      echo "Already replayed $start to $end, skipping"
      continue
    fi
    mkdir -p "$output_folder"

    temp_folder=$(mktemp -d)

    for ind in $(seq $start $end); do
      if [[ ! -f "${data_dir}/chunk_${ind}.tar.gz" ]]; then
        # copy csv to temp folder
        cp "${data_dir}/chunk_${ind}.csv" "$temp_folder"
        continue
      fi
      full_item_path="${data_dir}/chunk_${ind}.tar.gz"
      tar -xzf "$full_item_path" -C "$temp_folder"
    done

    trap cleanup SIGINT


    # create temp random output dir
    output_folder_temp=$(mktemp -d)

    ## call warmup
    ./replay.sh --file "./bin/warmup.csv" --output-dir "$output_folder_temp" --device /dev/nvme1n1

    # Loop through all files in the temp folder
    for file in "$temp_folder"/*; do
      ./replay.sh --file "$file" --output-dir "$output_folder" --device /dev/nvme1n1

      if [ $? -eq 130 ]; then
        exit 130
      fi
    done

    touch "$output_folder/done"
    rm -rf "$temp_folder"
  done
}


rescale_data() {
    local input_dir output_dir
    input_dir=$(parse_opt_req "input:i" "$@")
    output_dir=$(parse_opt_req "output:o" "$@")
    metric=$(parse_opt_req "metric:m" "$@")
    multiplier=$(parse_opt_req "multiplier:m" "$@")

    python -m clio.flashnet.cli.characteristic rescale \
        $input_dir \
        $output_dir \
        --metric $metric \
        --multiplier $multiplier
}
# +=================+
# |    __main__     |
# +=================+

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  [[ -z "${CLIO}" ]] && echo "CLIO env is not set" && exit 1
  # shellcheck disable=SC1091
  source "${CLIO}/util.sh"
  dorun "${@@Q}"
fi
