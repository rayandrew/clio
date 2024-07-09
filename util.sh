#!/usr/bin/env bash

_dorun_execute=0

blue='\033[0;34m'
green='\033[0;32m'
red='\033[0;31m'
yellow='\033[0;33m'
purple='\033[0;35m'
reset='\033[0m'

EXCLUDE_FUNCTIONS=(
  "dorun"
  "help"
  "load_task"
  "assert"
  "assert_ret"
  "log_info"
  "log_warn"
  "log_err"
  "log_success"
  "log_debug"
  "check_done"
  "mark_done"
  "check_env"
  "create_opts"
  "global_sanity_check"
  "global_setup_env"
  "echo_stderr"
  "parse_arg"
  "parse_opt"
  "parse_opt_req"
  "parse_opt_flag"
  "is_truthy"
  "is_falsy"
  "is_absolute_path"
  "canonicalize_path"
  "exec_report"
)
EXCLUDE_FUNCTIONS_STR=$(
  IFS="|"
  echo "${EXCLUDE_FUNCTIONS[*]}"
)

echo_stderr() {
  echo "$@" >&2
}

log_info() {
  echo_stderr -e "${blue}[I]${reset} $1"
}

log_warn() {
  echo_stderr -e "${yellow}[W]${reset} $1"
}

log_err() {
  echo_stderr -e "${red}[E]${reset} $1"
}

log_success() {
  echo_stderr -e "${green}[S]${reset} $1"
}

log_debug() {
  if [ "$DEBUG" -eq 1 ]; then
    echo_stderr -e "${purple}[D]${reset} $1"
  fi
}

assert() {
  # shellcheck disable=SC2128
  if [[ "$#" -ne 2 ]]; then
    log_err "Usage: $FUNCNAME <condition> <message>"
    exit 1
  fi

  if [[ "$1" = "false" ]] || [[ "$1" = "0" ]]; then
    log_err "Assertion failed: $2"
    exit 1
  fi
}

assert_ret() {
  if [[ "$?" -ne 0 ]]; then
    log_err "Assertion failed: $2"
    exit 1
  fi
}

is_truthy() {
  # shellcheck disable=SC2128
  assert "$#" "Usage: $FUNCNAME <value>"
  if [[ "$1" == "true" ]] || [[ "$1" == "1" ]] || [[ -n "$1" ]]; then
    echo 1
  else
    echo 0
  fi
}

is_falsy() {
  # shellcheck disable=SC2128
  assert "$#" "Usage: $FUNCNAME <value>"

  if [[ "$1" == "false" ]] || [[ "$1" == "0" ]] || [[ -z "$1" ]]; then
    echo 1
  else
    echo 0
  fi
}

help() {
  FUNCTIONS=$(compgen -A function | grep -e '^[^_]')

  # exclude EXCLUDE_FUNCTIONS from help
  mapfile -t FUNCTIONS < <(echo "${FUNCTIONS[@]}" | tr ' ' '\n' | grep -v -E "${EXCLUDE_FUNCTIONS_STR}")
  FUNCTIONS_STR=$(
    IFS="|"
    echo "${FUNCTIONS[*]}"
  )

  printf "Usage:\n\t$0 %s\n" "($FUNCTIONS_STR)"
}

load_task() {
  local task=$1
  shift

  # check if task contain ".sh", remove it
  if [[ $task == *".sh"* ]]; then
    task="${task//.sh/}"
  fi

  # check if task is separated by ".", convert to "/"
  if [[ $task == *"."* ]]; then
    task=$(echo "$task" | sed 's/\./\//g')
  fi

  if [ -f "$task.sh" ]; then
    # shellcheck disable=SC1090
    source "$task.sh"
    dorun $@
    # dorun "${*@Q}"
  else
    help
  fi
}

dorun() {
  # check if _dorun_execute is set
  if [ $_dorun_execute -eq 1 ]; then
    return
  fi

  # if [[ "$@" == *"--help"* ]] || [[ "$@" == *"-h"* ]] || [[ "$@" == *"help"* ]] || [[ "$@" == *"_dorun"* ]]; then
  # shellcheck disable=SC2199
  if [[ "$@" == *"--help"* ]] || [[ "$@" == *"-h"* ]] || [[ "$@" == *"help"* ]]; then
    help
    exit 0
  fi

  # check if task is defined inside EXCLUDE_FUNCTIONS
  # shellcheck disable=SC2199
  for exclude in "${EXCLUDE_FUNCTIONS[@]}"; do
    if [[ "$@" == "$exclude" ]]; then
      echo "Task $1 is defined inside EXCLUDE_FUNCTIONS"
      help
      exit 1
    fi
  done

  # shellcheck disable=SC2178
  FUNCTIONS=$(compgen -A function | grep -e '^[^_]')
  FUNCTIONS_STR=$(
    IFS="|"
    echo "${FUNCTIONS[*]}"
  )
  # echo "$FUNCTIONS_STR"

  # shellcheck disable=SC2068
  echo "$@"
  for task in $@; do
    # shellcheck disable=SC2199
    # if [[ "${FUNCTIONS[@]}" =~ $task ]]; then
    if [[ "${FUNCTIONS_STR}" =~ $task ]]; then
      # shellcheck disable=SC2124
      # args="$@"
      # args=${args//$task/}
      # chop first occurrence of $task in $args
      args=$(echo "$args" | awk -v task="$task" '{sub(task, "", $0); print}')
      # shellcheck disable=SC2086
      if [ "$TIMING" -eq 1 ]; then
        time {
          $task $args
          # $task "${args@Q}"
          TIMEFORMAT="==== Task \"$task\" took %R seconds ===="
        }
      else
        $task $args
        # $task "${args@Q}"
      fi

      exit_code=$?
      exit $exit_code
    else
      # shellcheck disable=SC2124
      # args="${*@Q}"
      args=("${@@Q}")
      # remove $task from $args
      # args=${args//$task/}
      # replace first occurence ONLY of $task in $args
      echo "BEFORE ${args[@]}"
      # args=$(echo "$args" | awk -v task="$task" '{sub(task, "", $0); print}')
      # remove from $args where $task is found
      # args=("${args[@]/$task}")
      # remove empty elements
      # args=("${args[@]// /}")
      cleaned_args=()
      for arg in "${args[@]}"; do
        if [[ "$arg" != *"$task"* ]]; then
          cleaned_args+=("$arg")
        fi
      done
      
      echo "AFTER ${cleaned_args[@]}"
      # shellcheck disable=SC2086
      load_task $task ${args@Q}
      exit_code=$?
      exit $exit_code
    fi
  done

  _dorun_execute=1

  if [ "$#" -gt 0 ]; then shift; fi
  [ "$#" -gt 0 ] || help
}

check_done() {
  if [ "$#" -eq 0 ]; then
    # shellcheck disable=SC2128
    log_err "Usage: $FUNCNAME <path>"
    exit 1
  fi
  local path=$1

  # path needs to be defined and directory
  if [ ! -d "$path" ]; then
    echo 0
    return
  fi

  # check if done file exists
  if [ -f "$path/done" ]; then
    echo 1
  else
    echo 0
  fi
}

check_done_ret() {
  if [ "$#" -eq 0 ]; then
    # shellcheck disable=SC2128
    log_err "Usage: $FUNCNAME <path> (msg)"
    exit 1
  fi

  local path=$1
  shift
  local msg="Already done"
  if [ "$#" -eq 1 ]; then
      msg=$1
  fi
  check=$(check_done "$path")
  if [ "$check" -eq 1 ]; then
    log_info "$msg"
    return 1
    # exit 0
  fi
  return 0
  # exit 1
}

mark_done() {
  if [ "$#" -eq 0 ]; then
    # shellcheck disable=SC2128
    log_err "Usage: $FUNCNAME <path>"
    exit 1
  fi
  local path=$1

  if [ ! -d "$path" ]; then
    exit 1
  fi

  log_info "Marking done"
  touch "$path/done"
}

check_env() {
  set +u
  local name=$1
  if [[ -z "${!name}" ]]; then
    # shellcheck disable=SC2128
    log_err "$FUNCNAME: Environment variable $name is not set"
    exit 1
  else
    exit 0
  fi
}

parse_arg() {
  # parse positional arguments until - or -- is found
  # return arr provided by reference
  local -n arr=$1
  shift
  local args=("$@")

  for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "-"* ]] || [[ "${args[$i]}" == "--"* ]]; then
      return
    fi
    arr+=("${args[$i]}")
  done
}

parse_opt() {
  if [[ "$#" -eq 0 ]]; then
    # shellcheck disable=SC2128
    log_err "Usage: $FUNCNAME <name> <...args>"
    exit 1
  fi

  # name will be long:short
  local name args long short
  name=$1
  shift
  # args=("$@")
  args=("${*@Q}")

  echo
  echo "Args"
  echo "  ${*@Q}"
  echo

  # line="${@@Q}"
  # echo $line

  long=$(echo "$name" | cut -d':' -f1)
  short=$(echo "$name" | cut -d':' -f2)

  while [[ "$#" -gt 0 ]]; do
    case $1 in
    "-$short" | "--$long")
     # echo "$2"
      shift

      args="$1"
      # check if args contain opening quote
      if [[ "$args" == \"* ]]; then
      echo "here"
        # remove opening quote
        args="${args#\"}"
        # parse until closing quote
        while [[ "$#" -gt 0 ]]; do
          if [[ "$1" == \"* ]]; then
            # remove closing quote
            args="$args ${1%\"}"
            break
          fi
          args="$args $1"
          shift
        done
      fi

      echo "${args# }"

      # parse until next flag
      # preserve the value even if it has quotes
      # args=""
      # while [[ "$#" -gt 0 ]]; do
      #   if [[ "$1" == "-"* ]] || [[ "$1" == "--"* ]]; then
      #     break
      #   fi
      #   args="$args $1"
      #   # echo "$1"
      #   shift
      # done
      # echo "${args# }"
        
      return
      ;;
    # accept with =
    "-$short="* | "--$long="*)
    # echo "here"

      args=("${1#*=}")
      
      # parse until next flag
      # preserve the value even if it has quotes
      while [[ "$#" -gt 0 ]]; do
        if [[ "$1" == "-"* ]] || [[ "$1" == "--"* ]]; then
          break
        fi
        args="$args $1"
        shift
      done

      echo "${args# }"

      return
      ;;
    *)
      shift
      ;;
    esac
  done
}

parse_opt_default() {
  if [[ "$#" -eq 0 ]]; then
    # shellcheck disable=SC2128
    log_err "Usage: $FUNCNAME <name> <default> <...args>"
    exit 1
  fi

  local name default result
  name=$1
  default=$2
  shift 2

  result=$(parse_opt "$name" "$@")
  if [[ -z "$result" ]]; then
    echo "$default"
  else
    echo "$result"
  fi
}

parse_opt_req() {
  if [[ "$#" -eq 0 ]]; then
    # shellcheck disable=SC2128
    log_err "Usage: $FUNCNAME <name> <...args>"
    exit 1
  fi

  local name cut_name result
  name=$1

  cut_name=$(echo "$name" | cut -d':' -f1)
  shift
  result=$(parse_opt "$name" "$@")
  if [[ -z "$result" ]]; then
    # shellcheck disable=SC2128
    log_err "$FUNCNAME: Option \"$cut_name\" is required"
    exit 1
  fi
  echo "$result"
}

parse_opt_flag() {
  if [[ "$#" -eq 0 ]]; then
    # shellcheck disable=SC2128
    log_err "Usage: $FUNCNAME <name> <...args>"
    exit 1
  fi

  # name will be long:short
  local name args long short
  name=$1
  shift
  args=("$@")

  long=$(echo "$name" | cut -d':' -f1)
  short=$(echo "$name" | cut -d':' -f2)

  while [[ "$#" -gt 0 ]]; do
    case $1 in
    "-$short" | "--$long")
      echo "1"
      return
      ;;
    *)
      shift
      ;;
    esac
  done
}

is_absolute_path() {
  if [[ "$#" -ne 1 ]]; then
    # shellcheck disable=SC2128
    log_err "Usage: $FUNCNAME <path>"
    exit 1
  fi

  local path=$1
  if [[ "$path" == /* ]]; then
    echo 1
  else
    echo 0
  fi
}

canonicalize_path() {
  if [[ "$#" -ne 1 ]]; then
    # shellcheck disable=SC2128
    log_err "Usage: $FUNCNAME <path>"
    exit 1
  fi

  realpath -m "$1"
}

exec_report() {
  local cmd=("$@")
  # shellcheck disable=SC2145
  log_info "Executing: ${cmd[*]}"
  "$@"
}

global_sanity_check() {
  assert_ret "$(check_env MAMBA_ROOT_PREFIX)" "MAMBA_ROOT_PREFIX is not set"
  assert_ret "$(check_env CLIO)" "CLIO is not set"
}

global_setup_env() {
  export ENV_NAME="clio"
  export PATH="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}/bin:${PATH}"
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
  export TF_CPP_MIN_LOG_LEVEL="3"
  export CUDA=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export _TYPER_STANDARD_TRACEBACK="1"
  export TQDM_DISABLE="0"
  export LOG_LEVEL="INFO"
  export DEBUG=${DEBUG:-0}
  export TIMING=${TIMING:-1}
}

global_sanity_check
global_setup_env
