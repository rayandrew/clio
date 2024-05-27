#!/bin/bash

# Specify the path to the SSD drive or device you want to test
ssd_device="/dev/$1"  # Replace with the actual device name (e.g., /dev/sda)

# Specify the path to the output files
output_read="tmp_ssd_random_read.txt"
output_write="tmp_ssd_random_write.txt"

# Specify FIO test parameters for random read
fio_test_read="[ssd_random_read]"
fio_runtime="20s"      # Test duration (e.g., 60 seconds)
fio_size="1G"          # Size of the test file (e.g., 1 gigabyte)
fio_direct=1           # Use direct I/O
fio_ioengine="libaio"
fio_numjobs=1
fio_rw="randread"      # Test type (random read)
fio_blocksize="4k"     # I/O block size (e.g., 4KB)

# Specify FIO test parameters for random write
fio_test_write="[ssd_random_write]"
fio_rw_write="randwrite"  # Test type (random write)

# Create an FIO job file for random read
cat <<EOF > fio_job_file_read
$fio_test_read
filename=$ssd_device
rw=$fio_rw
size=$fio_size
direct=$fio_direct
ioengine=$fio_ioengine
runtime=$fio_runtime
numjobs=$fio_numjobs
blocksize=$fio_blocksize
EOF

# Create an FIO job file for random write
cat <<EOF > fio_job_file_write
$fio_test_write
filename=$ssd_device
rw=$fio_rw_write
size=$fio_size
direct=$fio_direct
ioengine=$fio_ioengine
runtime=$fio_runtime
numjobs=$fio_numjobs
blocksize=$fio_blocksize
EOF

# Run the FIO tests
fio fio_job_file_read --output=$output_read
fio fio_job_file_write --output=$output_write

# Display test results
echo "Random Read Test Results:"
cat $output_read

echo "Random Write Test Results:"
cat $output_write

# Clean up
rm fio_job_file_read fio_job_file_write
