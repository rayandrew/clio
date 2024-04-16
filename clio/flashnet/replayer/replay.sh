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
      ;;
    -o|--output_dir)
      output_dir="$2"
      ;;
    -r|--replayer)
      replayer="$2"
      ;;
    *)
      printf "ERROR: Invalid argument. \n(sample: ./replay.sh --device /dev/nvme0n1 --dir \$FLASHNET/data/trace_raw/ --pattern "*cut*trace" --output_dir \$FLASHNET/data/trace_profile/)\n"
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
    stats_path=$(generate_stats_path)
    sudo $IO_REPLAYER_PATH $device $file $output_path 
    echo "output replayed trace : ${output_path}"
    echo "         output stats : ${stats_path}"
    chown -R $user "$output_dir/$dev_name"
    chown $user "$output_path"
    chown $user "$stats_path"
}

echo "DEVICE=$device, DIR=$dir, PATTERN=$pattern, OUTPUT_DIR=$output_dir"

# translate the pattern to glob results


if [[ $device && $dir && $pattern && $output_dir ]]; then
    python -c 'from glob import glob; print(glob("'$dir'/'$pattern'"))' | while read -r file; do
        if [[ -f $file ]]; then # check this file                    
            echo $file
            # replay_file
        fi
    done

    # Iterate through the files in dir 
    # for file in ${dir}/*; do
    #     # echo "file = $file"
    #     if [[ -f $file ]]; then # check this file                    
    #         # check whether it satisfy the pattern
    #         python -c 'from glob import glob; print(glob("'$pattern'"))' | grep -q $file
    #         # if ../../src/commonutils/pattern_checker.py -pattern ${pattern} -file ${file} | grep -q 'True'; then
    #             replay_file
    #         fi
    #     fi
    # done
elif [[ $device && $file && $output_dir ]]; then
    # replay_file 
fi