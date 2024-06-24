
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

# labeling + feature engineering
postprocess() {
    local input_file output_path
    input_dir=$(parse_opt_req "input:i" "$@")
    output_dir=$(parse_opt_req "output:o" "$@")

    python -m clio.flashnet.cli.characteristic generate_v2 \
        $input_dir \
        --output $output_dir \
        --relabel
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
      # echo "Replaying $ind"
      full_item_path="${data_dir}/chunk_${ind}.tar.gz"
      tar -xzf "$full_item_path" -C "$temp_folder"
    done

    trap cleanup SIGINT

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


# +=================+
# |    __main__     |
# +=================+

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  [[ -z "${CLIO}" ]] && echo "CLIO env is not set" && exit 1
  # shellcheck disable=SC1091
  source "${CLIO}/util.sh"
  dorun "$@"
fi
