#!/usr/bin/env bash

date=$(date '+%Y_%b_%d')
hostname=$(hostname)
output_file="specs_collection/$hostname.specs.$date.txt" 
echo "Generating output at $output_file"
sudo ./collect_specs.sh > $output_file
sudo chown $USER:$USER $output_file
rm tmp_ssd_random_*.txt