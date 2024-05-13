#!/usr/bin/env bash

set -e

# wget "https://raw.githubusercontent.com/xonixx/makesure/main/makesure?token=$(date +%s)" -Omakesure && \
# chmod +x makesure && echo "makesure $(./makesure -v) installed"

cwd=$(pwd)
pushd /tmp
wget https://github.com/TekWizely/run/releases/download/v0.11.2/run_0.11.2_linux_amd64.tar.gz
tar -xf run_0.11.2_linux_amd64.tar.gz
if [ -f run ]; then
    chmod +x run
fi
mv run $cwd
popd

# micromamba env create --name clio --file env.yaml -y

# pip install --upgrade fitsne