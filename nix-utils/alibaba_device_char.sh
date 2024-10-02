#!/bin/bash

# Configuration
device_ids_csv="/home/cc/clio/nix-utils/alibaba_device_rw-device_counts.csv"  # Path to the CSV file with device IDs
duckdb_command="/mnt/nvme0n1/duckdb"
db_path="/mnt/nvme0n1/iotrace_master"
output_base_dir="/mnt/nvme0n1/alibaba_block_traces_2020"
split_output_base_dir="./runs/raw/alibaba_loop/split"
characteristics_output_base_dir="./runs/raw/alibaba_loop/characteristic"
error_log="./error.log"  # Path to the error log file

# Read device IDs from CSV file (rows 2 to 101)
device_ids=($(awk -F, 'NR>=5 && NR<=106 {print $1}' "$device_ids_csv"))

# Loop through each device_id
for device_id in "${device_ids[@]}"; do
    csv_output_dir="${output_base_dir}/device_${device_id}_csv_glob"
    
    echo "Processing device_id: ${device_id}"
    
    # Check if output directory already exists
    if [ -f "${characteristics_output_base_dir}/${device_id}/10m/characteristic.csv" ]; then
        echo "Output characteristic 10m already exists. Skipping..."
        echo "Deleting ${csv_output_dir}"
        rm -rf ${csv_output_dir}
        continue
    fi

    if [ ! -d "${csv_output_dir}" ]; then
        echo "Running duckdb commands SHOW TABLES;
COPY (SELECT * FROM alibaba_whole WHERE device_id = '${device_id}') 
TO '${csv_output_dir}' 
(FORMAT CSV, per_thread_output True);"

    ${duckdb_command} --readonly ${db_path} <<EOF 2>>${error_log}
SHOW TABLES;
COPY (SELECT * FROM alibaba_whole WHERE device_id = '${device_id}') 
TO '${csv_output_dir}' 
(FORMAT CSV, per_thread_output True);
EOF
    fi
    

    if [ $? -ne 0 ]; then
        echo "Error occurred while processing device_id: ${device_id}. Check ${error_log} for details."
        continue
    fi

    # Run the split command
    split_output_dir="${split_output_base_dir}/${device_id}/"
    echo "./r s/raw/alibaba split --input ${csv_output_dir} --output ${split_output_dir}"
    ./r s/raw/alibaba split --input ${csv_output_dir} --output ${split_output_dir}

    # Calculate characteristics
    characteristics_output_dir_10m="${characteristics_output_base_dir}/${device_id}/10m"
    ./r s/processing calc_raw_characteristic --input ${split_output_dir}/1m --output ${characteristics_output_dir_10m} --window 10m 

    # characteristics_output_dir_1m="${characteristics_output_base_dir}/${device_id}/1m"
    # ./r s/processing calc_raw_characteristic --input ${split_output_dir}/1m --output ${characteristics_output_dir_1m} --window 1m 

    # If outputs are all good, clean up intermediate files, check for existence of output files
    # check for file characteristics.csv in the output directory
    if [ -f "${characteristics_output_dir_10m}/characteristic.csv" ]; 
    then
        echo "Output files for device_id: ${device_id} are ready."
        echo "Cleaning up intermediate files for device_id: ${device_id}"
        rm -rf ${split_output_dir}
        rm -rf ${csv_output_dir}
    else
        echo "Error occurred while processing device_id: ${device_id}. Check ${error_log} for details."
    fi

    echo "Finished processing device_id: ${device_id}"
done

echo "All devices processed successfully."
