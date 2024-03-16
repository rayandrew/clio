#!/usr/bin/env bash

set -e

wget "https://raw.githubusercontent.com/xonixx/makesure/main/makesure?token=$(date +%s)" -Omakesure && \
chmod +x makesure && echo "makesure $(./makesure -v) installed"

micromamba env create --name clio --file env.yaml -y

# pip install --upgrade fitsne