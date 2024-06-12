#!/bin/bash


while [ $# -gt 0 ]; do
  case "$1" in
    -u|--user)
      user="$2" # to regain the access from the sudo user
      ;;
    --device)
      device="$2"
      ;;
    -d|--dir)
      dir="$2"
      ;;
    -f|--file)
      file="$2"
      ;;
    -p|--pattern)
      pattern="$2"
      pattern=${pattern//\"/}
      ;;
    -o|--output-dir)
      output_dir="$2"
      ;;
    -r|--replayer)
      replayer="$2"
      ;;
    -f|--force)
      force=true
      ;;
    *)
      # printf "ERROR: Invalid argument. \n(sample: ./replay.sh --device /dev/nvme0n1 --dir \$FLASHNET/data/trace_raw/ --pattern "*cut*trace" --output_dir \$FLASHNET/data/trace_profile/)\n"
      printf "ERROR: Invalid argument. \n"
      printf "  Invalid argument: $1\n"
      printf "  Usage: ./replay.sh --device /dev/nvme0n1 --dir \$FLASHNET/data/trace_raw/ --pattern \"*cut*trace\" --output-dir \$FLASHNET/data/trace_profile/ --user \$USER\n"
      exit 1
  esac
  shift
  shift
done


CURR_DIR=$(pwd)

if [[ ! -z $replayer ]]; then
    IO_REPLAYER_PATH="$replayer"
else
    IO_REPLAYER_PATH="$CURR_DIR/io_replayer"
fi

# find absolute path of io_replayer
IO_REPLAYER_PATH=$(realpath $IO_REPLAYER_PATH)

echo "Finding io_replayer at $IO_REPLAYER_PATH"

if [[ ! -f $IO_REPLAYER_PATH ]]; then
    echo "ERROR: io_replayer not found in the $IO_REPLAYER_PATH"
    exit 1
fi

dev_name=${device//"/dev/"/} # remove the "/dev/" -> we just need the device name
# echo "user: ${user}"

function generate_output_path()
{
    filename=$(basename ${file})
    echo "$output_dir/$dev_name/$filename"
}

function generate_stats_path()
{
    filename=$(basename ${file})
    echo "$output_dir/$dev_name/$filename.stats"
}

function replay_file()
{
    echo ""
    echo "Replaying on ${dev_name} : ${file}"
    output_path=$(generate_output_path)
    # stats_path=$(generate_stats_path)
    sudo $IO_REPLAYER_PATH $device $file $output_path 
    echo "output replayed trace : ${output_path}"
    # echo "         output stats : ${stats_path}"
    if [[ ! -z $user ]]; then
      sudo chown -R $user "$output_dir/$dev_name"
      sudo chown $user "$output_path"
      # sudo chown $user "$stats_path"
    fi
}

# echo "DEVICE=$device, DIR=$dir, PATTERN=$pattern, OUTPUT_DIR=$output_dir"

if [[ $device && $dir && $pattern && $output_dir ]]; then
    python3 -c '
from glob import glob

for p in glob("'$dir'/'$pattern'"):
    print(p)
' | sort -V | while read -r file; do
        if [[ -f $file ]]; then # check this file                    
            # echo $file
            # check if the output file already exists
            output_path=$(generate_output_path)
            if [[ -f $output_path && -z $force ]]; then
                echo "Output file already exists: $output_path"
                continue
            fi
            replay_file
            sleep 1
        fi
    done
elif [[ $device && $file && $output_dir ]]; then
  # echo "Replaying on ${dev_name} : ${file}"
  replay_file 
fi

