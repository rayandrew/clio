# Trace Characteristic + Editor

## Prerequisites

- Rust
- GNU Parallel

## Tencent

Download Tencent Dataset from SNIA [here](http://iotta.snia.org/traces/parallel/27917?n=10&page=1)

### Information

Copied from SNIA Tencent README

```
There are 216 I/O traces from a warehouse (also called a failure domain) of a production cloud block storage system (CBS). The traces are I/O requests from 5584 cloud virtual volumes (CVVs) for ten days (from Oct. 1st to Oct. 10th, 2018). The I/O requests from the CVVs are mapped and redirected to a storage cluster consists of 40 storage nodes (i.e., disks). 

OSCA: An Online-Model Based Cache Allocation Scheme in Cloud Block Storage Systems

Each trace file is named as <date>.tgz.

The date format consists of year(4 char.), month(2 char.), day(2 char.), and hour(2 char.).

Here is an example file name.
2018-10-01-01.tgz
  - year : 2018
  - month : 10 (Oct.)
  - day : 1st
  - hour : 1 AM

=== I/O trace file format ===

Timestamp,Offset,Size,IOType,VolumeID

  - Timestamp is the time the I/O was issued.
    The timestamp is given as a Unix time (seconds since 1/1/1970).       
  - Offset is the starting offset of the I/O in sectors from the start of the logical virtual volume.
    1 sector = 512 bytes
  - Size is the transfer size of the I/O request in sectors.
  - IOType is "Read(0)", "Write(1)".
  - VolumeID is the ID number of a CVV.

=== Some suggestions on the use of the traces ===
In addition to using these traces to examine the issues discussed in the ATC '20 paper, they can be used to analyze:

	(1) the CBS access patterns of CVVs by dividing traces for each CVV using the VolumeID.
	(2) the architecture optimization of CBS by simulating the mapping of I/O requests to storage nodes.
```

- Count volume in raw traces

```bash
./r
```

- Join the device/volume characteristics

```bash
./target/release/tencent_device_joiner ./devices/
```

- Filter device

```bash
# change VOLUME=xxxx inside the filter_device.sh
./scripts/tencent/filter_device.sh # will output to ./filtered_devices/xxxx
```

- Split trace

```bash
# 
./scripts/tencent/split
```

### Pipeline

- Get device/volume
- Pick the top used device/volume
- Filter the device/volume

