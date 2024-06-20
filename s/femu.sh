#!/usr/bin/env bash

set -e

########################################################
############# Run the script in this order #############
# 1. setup 
# 2. compile 
# 3. download_image 
# 3.1. spawn
# 4. post_vm_setup
#   4.1 Please remove sudo password for femu user
# 5. vm_install_env
# 6. prepare_replayer
########################################################

__env__() {
  export HOME="/home/cc"
  export CC=cc
  export CXX=c++
  export CLIO="/home/cc/clio"
  export LOCAL_REPLAYER_PATH="${CLIO}/clio/flashnet/replayer"
  # export SSH_DIR="${HOME}/.ssh/id_rsa"
  export FEMU_REPLAYER_PATH="/home/femu/replayer"
}

setup() {
    if [[ ! -d femu ]]; then
      git clone https://github.com/vtess/femu.git
    fi
    pushd femu
    mkdir build-femu
    pushd build-femu
    cp ../femu-scripts/femu-copy-scripts.sh .
    ./femu-copy-scripts.sh .
    sudo ./pkgdep.sh
    popd
    popd
}

compile() {
  pushd femu
  pushd build-femu
  ./femu-compile.sh
  popd
  popd
}

download_image() {
    mkdir -p .tmp/images
    pushd .tmp/images
    rm -f femu-vm.tar.xz
    wget http://people.cs.uchicago.edu/~huaicheng/femu/femu-vm.tar.xz
    tar xJvf femu-vm.tar.xz
    rm -f femu-vm.tar.xz
    popd
}

post_vm_setup() {
  ssh-copy-id -p 8080 femu@localhost
  log_info "Now log into the VM using the following command"
  log_info "  ssh -p 8080 femu@localhost"
  log_info "  Run 'sudo visudo'"
  log_info "  And add 'femu ALL=(ALL) NOPASSWD: ALL' at the end of the file"
  log_info "  Then type CTRL+X, y, and ENTER"
}

vm_install_env() {
  ssh -v -p  8080 femu@localhost << EOF
sudo apt-get update
sudo apt-get install -y python3 python3-pip
EOF
}

spawn() {
    local img_dir 
    img_dir=$(parse_opt_default "image-dir:i" ".tmp/images" "$@")
    img_dir=$(canonicalize_path "$img_dir")

    pushd femu/build-femu

    ##### run-blackbox.sh ---------------------------------------------
    
    # Virtual machine disk image
    OSIMGF="$img_dir/u20s.qcow2"

    # Configurable SSD Controller layout parameters (must be power of 2)
    secsz=512 # sector size in bytes
    secs_per_pg=8 # number of sectors in a flash page
    pgs_per_blk=256 # number of pages per flash block
    blks_per_pl=256 # number of blocks per plane
    pls_per_lun=1 # keep it at one, no multiplanes support
    luns_per_ch=8 # number of chips per channel
    nchs=8 # number of channels
    ssd_size=12288 # in megabytes, if you change the above layout parameters, make sure you manually recalculate the ssd size and modify it here, please consider a default 25% overprovisioning ratio.

    # Latency in nanoseconds
    pg_rd_lat=40000 # page read latency
    pg_wr_lat=200000 # page write latency
    blk_er_lat=2000000 # block erase latency
    ch_xfer_lat=0 # channel transfer time, ignored for now

    # GC Threshold (1-100)
    gc_thres_pcent=75
    gc_thres_pcent_high=95

    #-----------------------------------------------------------------------

    #Compose the entire FEMU BBSSD command line options
    FEMU_OPTIONS="-device femu"
    FEMU_OPTIONS=${FEMU_OPTIONS}",devsz_mb=${ssd_size}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",namespaces=1"
    FEMU_OPTIONS=${FEMU_OPTIONS}",femu_mode=1"
    FEMU_OPTIONS=${FEMU_OPTIONS}",secsz=${secsz}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",secs_per_pg=${secs_per_pg}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",pgs_per_blk=${pgs_per_blk}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",blks_per_pl=${blks_per_pl}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",pls_per_lun=${pls_per_lun}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",luns_per_ch=${luns_per_ch}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",nchs=${nchs}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",pg_rd_lat=${pg_rd_lat}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",pg_wr_lat=${pg_wr_lat}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",blk_er_lat=${blk_er_lat}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",ch_xfer_lat=${ch_xfer_lat}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",gc_thres_pcent=${gc_thres_pcent}"
    FEMU_OPTIONS=${FEMU_OPTIONS}",gc_thres_pcent_high=${gc_thres_pcent_high}"

    log_info "FEMU OPTIONS"
    log_info "${FEMU_OPTIONS}"

    if [[ ! -e "$OSIMGF" ]]; then
      echo ""
      echo "VM disk image couldn't be found ..."
      echo "Please prepare a usable VM image and place it as $OSIMGF"
      echo "Once VM disk image is ready, please rerun this script again"
      echo ""
      exit
    fi

    sudo ./qemu-system-x86_64 \
        -name "FEMU-BBSSD-VM" \
        -enable-kvm \
        -cpu host \
        -smp 4 \
        -m 4G \
        -device virtio-scsi-pci,id=scsi0 \
        -device scsi-hd,drive=hd0 \
        -drive file="$OSIMGF",if=none,aio=native,cache=none,format=qcow2,id=hd0 \
        ${FEMU_OPTIONS} \
        -net user,hostfwd=tcp::8080-:22 \
        -net nic,model=virtio \
        -nographic \
        -qmp unix:./qmp-sock,server,nowait 2>&1 | tee log

    # -------------------------------------------------------------------

    popd
}

prepare_replayer() {
  rsync -Pavr -e "ssh -p 8080" "$LOCAL_REPLAYER_PATH" femu@localhost:/home/femu
  # shellcheck disable=SC2087
  ssh -p 8080 femu@localhost << EOF
  cd ${FEMU_REPLAYER_PATH}
  gcc replayer.c -o io_replayer -lpthread
  echo "Replayer compiled"
EOF
}

## ./s/femu.sh replay_trace --trace ./runs/raw/tencent/split/123/ --output ./replayed/dir
replay_trace() {
  local trace_path is_trace_dir pattern
  trace_path=$(parse_opt_req "trace:t" "$@")
  pattern=$(parse_opt_default "pattern:p" "*" "$@")
  trace_path=$(canonicalize_path "$trace_path")
  
  if [[ -d "$trace_path" ]]; then
    is_trace_dir=true
  else
    is_trace_dir=false
  fi

  if [[ "$is_trace_dir" == true ]] && [[ -z "$pattern" ]]; then
      log_info "Pattern is required for directory"
      exit 1
  fi

  base_trace_path=$(basename "$trace_path")
  base_trace_path_without_ext=$(echo "$base_trace_path" | cut -f 1 -d '.')
  echo "base_trace_path: $base_trace_path_without_ext"

  local femu_input_path femu_output_path output_path
  femu_input_path="/home/femu/trace/$base_trace_path_without_ext"
  femu_output_path="/home/femu/trace/replay/$base_trace_path_without_ext"

  output_path=$(parse_opt_default "output:o" "." "$@")
  output_path=$(canonicalize_path "$output_path")
  mkdir -p "$output_path"

  ssh -p 8080 femu@localhost "mkdir -p /home/femu/trace"
  rsync -Pavr -e "ssh -p 8080" "$trace_path" femu@localhost:/home/femu/trace
  # shellcheck disable=SC2087
  ssh -p 8080 femu@localhost << EOF
cd /home/femu/trace

if [[ "$is_trace_dir" == true ]]; then
  pushd $base_trace_path
  for f in *; do
    if [[ "\$f" == *.tar.gz ]]; then
      echo "Extracting \$f"
      tar -xvf "\$f"
      rm -f "\$f"
    elif [[ "\$f" == *.gz ]]; then
      echo "Extracting \$f"
      gunzip "\$f"
      rm -f "\$f"
    fi
  done
  popd
else
  # if tar.gz file, extract
  if [[ "$base_trace_path" == *.tar.gz ]]; then
    echo "Extracting $base_trace_path"
    tar -xvf "$base_trace_path"
    rm -f "$base_trace_path"
  elif [[ "$base_trace_path" == *.gz ]]; then
    echo "Extracting $base_trace_path"
    gunzip "$base_trace_path"
    rm -f "$base_trace_path"
  fi
fi

echo "Running replayer for $base_trace_path_without_ext"
OUTPUT_PATH=$femu_output_path
rm -rf \$OUTPUT_PATH
mkdir -p \$OUTPUT_PATH

INPUT_PATH=$femu_input_path

pushd ${FEMU_REPLAYER_PATH}
if [[ "$is_trace_dir" == true ]]; then
  ./replay.sh --dir \$INPUT_PATH --output-dir \$OUTPUT_PATH --device /dev/nvme0n1 --user femu --pattern \"$pattern\"
else
  ./replay.sh --file \$INPUT_PATH --output-dir \$OUTPUT_PATH --device /dev/nvme0n1 --user femu
fi
popd
EOF

  rsync -Pavr -e "ssh -p 8080" femu@localhost:"$femu_output_path/nvme0n1/*" "$output_path"
  echo "Output is saved at $output_path"
  # shellcheck disable=SC2087
  ssh -p 8080 femu@localhost << EOF
  rm -rf $femu_input_path
  rm -rf $femu_output_path
EOF
}

# ./s/femu.sh replay_list --range-list "/home/cc/clio/test/iops/picked_drifts.csv" --data-dir "/home/cc/clio/runs/raw/tencent/split/1063" --output "./data/test"
replay_list() {
  local range_list data_dir output_dir
  range_list=$(parse_opt_req "range-list:r" "$@")
  data_dir=$(parse_opt_req "data-dir:d" "$@")
  output_dir=$(parse_opt_req "output:o" "$@")

  while IFS=, read -r start end type should_replay; do
    if [[ "$should_replay" != "y" ]]; then
      continue
    fi
    echo "Replaying: \n Start: $start, End: $end, Type: $type"
    output_folder="${output_dir}/${type}/${start}_${end}/raw/"
    if [[ -f $output_folder/done ]]; then
      echo "Already replayed $start to $end, skipping"
      continue
    fi
    for i in $(seq $start $end); do
      echo "Replaying $i"
      mkdir -p $output_folder
      ./s/femu.sh replay_trace --trace "${data_dir}/chunk_${i}.tar.gz" --output $output_folder
      touch $output_folder/done
    done
  done < "$range_list"
}
__env__

# +=================+
# |    __main__     |
# +=================+

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  [[ -z "${CLIO}" ]] && echo "CLIO env is not set" && exit 1
  # shellcheck disable=SC1091
  source "${CLIO}/util.sh"
  dorun "$@"
fi