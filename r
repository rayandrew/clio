#!/usr/bin/env bash

set -e

export PATH="${PATH}:${CLIO}/bin:${CLIO}/utils"

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

# +=================+
# |    __main__     |
# +=================+

dorun "$@"

