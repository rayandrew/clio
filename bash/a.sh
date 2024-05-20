#!/usr/bin/env bash

hello() {
    local arg
    parse_arg arg "$@"
    log_info "Hello, World!"
    for i in "${!arg[@]}"; do
        echo "$i: ${arg[$i]}"
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