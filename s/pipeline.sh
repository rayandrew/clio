#!/usr/bin/env bash

labeling() {
    local data
    data=$(parse_opt_req "data:d" "$@")
    log_info "Labeling the dataset: $data"
}

feature_engineering() {
    log_info "Feature engineering"
}

chain() {
    local data
    data=$(parse_opt_req "data:d" "$@")
    labeling --data="$data"
    feature_engineering
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
