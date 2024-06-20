#!/usr/bin/env bash


initial_only() {
  local data_dir output_dir cuda epochs
  data_dir=$(parse_opt_req "data:d" "$@")
  output_dir=$(parse_opt_req "output:o" "$@")
  cuda=$(parse_opt_default "cuda:c" 0 "$@")
  epochs=$(parse_opt_default "epochs:e" 20 "$@")

  if [[ -f "$output_dir/done" ]]; then
    echo "Already ran the experiment, skipping"
    exit 0
  fi
  echo "Running initial-only experiment"
    python -m clio.flashnet.cli.exp.single.run initial-only \
      $data_dir \
      --output $output_dir/single.initial-only \
      --cuda $cuda \
      --epochs $epochs
  echo "Experiment done" > "$output/done"
}

plot_exp() {
    local input_dir output_dir
    input_dir=$(parse_opt_req "input:i" "$@")
    output_dir=$(parse_opt_req "output:o" "$@")

    python -m clio.flashnet.cli.exp.analysis \
      $input_dir \
      --output $output_dir
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
