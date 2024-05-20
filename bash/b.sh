#!/usr/bin/env bash

w() {
    echo "hi from w"
}

we() {
    echo "hi from we"
}

wa() {
    echo "hi from wa"
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