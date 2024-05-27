# Trace Characteristic + Editor

## Prerequisites

- Rust
- GNU Parallel

## Tencent

Download Tencent Dataset from SNIA [here](http://iotta.snia.org/traces/parallel/27917?n=10&page=1)

- Find device/volume characteristics

```bash
./scripts/tencent/get_device.sh # will output to ./devices
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

