#!/bin/bash

echo "Hostname:"
hostname

echo "Date and Time:"
date

echo "Processor Specifications:"
lscpu

echo -e "\nOperating System Information:"
uname -a

echo -e "\nMemory Information:"
free -h


echo -e "\nSSD/HDD Information from each drive in lsblk:"
lsblk
lsblk -d -o name,rota
lsblk -o NAME | grep -E "^sd"
lsblk -o NAME | grep -E "^nvme"  

grep_result=$(lsblk -o NAME | grep -E "^nvme")
if [ ! -z "$grep_result" ]; then
    # Iterate over the matched lines
    while IFS= read -r line; do
        echo ""
        echo "====================================================================="
        echo "Checking the manufacturer and model of: $line"
        sudo smartctl -a /dev/$line | grep -E "Model|Vendor"
        # Perform actions on each matched line
        echo "Checking the speed of: $line"
        ./ssd_speed.sh $line
        # Add your additional processing here
    done <<< "$grep_result"
fi

grep_result=$(lsblk -o NAME | grep -E "^sd")
if [ ! -z "$grep_result" ]; then
    # Iterate over the matched lines
    while IFS= read -r line; do
        echo ""
        echo "====================================================================="
        echo "Checking the manufacturer and model of: $line"
        sudo smartctl -a /dev/$line | grep -E "Model|Vendor"
        # Perform actions on each matched line
        echo "Checking the speed of: $line"
        ./ssd_speed.sh $line
        # Add your additional processing here
    done <<< "$grep_result"
fi