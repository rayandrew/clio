
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
# +=================+
# |    __main__     |
# +=================+

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  [[ -z "${CLIO}" ]] && echo "CLIO env is not set" && exit 1
  # shellcheck disable=SC1091
  source "${CLIO}/util.sh"
  dorun "$@"
fi
