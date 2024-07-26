# CLIO

## Environment

Add this line in your `.zshrc/.bashrc` or other shell-compatible config

```bash
export CLIO=/path/to/this/repository
```

### Dependencies

- [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (Conda also works)
- Latex (for generating PDFs and combining plots)

### Installation

```bash
micromamba env create --name clio --file env.yaml
```

### Activation

```bash
micromamba activate clio
```

### Trace Utils

```bash
sudo apt-get install libtool automake autotools-dev autoconf
```

### gnuplot

```bash
sudo apt-get install gnuplot-qt ghostscript parallel
```

### direnv
Download direnv https://direnv.net/docs/installation.html.

### Getting started
[TODO] configurable windows, currently only tested on 1m windows only, currently working on variable window period

0. Setup FEMU, an SSD emulator. Run the following commands.
- ./r s/femu setup
- ./r s/femu compile
- ./r s/femu download_image
- ./r s/femu spawn
- ./r s/femu post_vm_setup -> Disable sudo user password, login to VM. See prompt
- ./r s/femu prepare_replayer

1. Download /runs folder from chameleon UC object store. First, get your swift credentials by following the guide below https://chameleoncloud.readthedocs.io/en/latest/technical/cli.html#cli-rc-script. You will then have a file containing the needed credentials to execute this
```bash
swift --os-auth-type v3applicationcredential --os-application-credential-id da8eb9b3943c452fa4183fad9d16e58c --os-application-credential-secret AUp1cJZ9ZHiUnAaPuXE8V55NFZ3Cu2Us4DQzXN0wQZIAvFQJ0cDURAy7NLzjckwfefAQsSDbiFU92JvN0cfs0A download ray-stor -p mnt
```

This will download:
- Characteristic, a csv containing metrics like IOPS, IAT, etc with windows split every 1m, 10m, 1h, etc. 
- Split, I/O traces split into 1 minute intervals

2. We will be using the characteristic file to get an idea of where drift might be, using a target metric. We can run a script that will produce a list of drifts like so, in this command, using the IOPS characteristic of device 1063 to find drift in 1 minute windows: 
`./r s/processing compile_and_get_drifts -o runs/exp/tencent/1063/1m -i ./runs/raw/tencent/characteristic/1063/1m/characteristic.csv -m iops --stability-threshold 15 --drift-threshold 50 --group-threshold 250 --group-offset 50 --rolling-window 10`

3. You can also plot the drifts from the previous step by the following command
`./r s/processing plot_drifts -p ./runs/exp/tencent/1063/1m/iops/data -o ./runs/exp/tencent/1063/1m/iops/drift_viz`

3. The csv will give a lot of potential drifts. We need to select a subset to replay. Set the column in to_be_picked_drifts.csv to 'y' if you want that drift replayed. Note that using 1 minute windows, replaying will take approximately ~1m too. So replaying windows from idx 100-200 will take ~100 minutes.

To make a line plot of the selected_drifts, run this command

`./r line_plot_selected_drift --range-list ./runs/exp/tencent/1063/1m/iops/selected_drifts.csv --char ./runs/raw/tencent/characteristic/1063/1m/characteristic.csv --output ./runs/exp/tencent/1063/1m/iops/line_plot_selected`

4. Replay the chunks marked by 'y' in the csv file by running. Range-list is a csv to read. This will loop through the CSV file, get the rows marked by 'y', then replay chunks from start to finish in FEMU (an SSD emulator). Data_dir should point to the folder containing files from chunks_0 to chunks_XXX.
`./r s/femu replay_list --range-list "./runs/exp/tencent/1063/1m/iops/selected_drifts.csv" --data-dir "./runs/raw/tencent/split/1282" --output "./runs/exp/tencent/1282/1m/iops/replayed" --time-split 1m`

* You can plot the tail latency of the replayed data by running this command, which will glob everything in the /reeplayed directory. Note that this will only plot drifts that have finished replaying (contains a done file)
`./r cdf_concat_from_replay_dir_glob -d runs/exp/tencent/1063/1m/iops/replayed/ -o runs/exp/tencent/1063/1m/iops/replayed_cdf -f`

5. Once done replaying, we can label and feature engineer everything in the replayed folder. This will output files that will be used to train our models.
`./r s/processing postprocess --input ./runs/exp/tencent/1063/1m/iops/replayed/gradual/105_117/raw --output ./runs/exp/tencent/1063/1m/iops/processed/gradual/105_117/`

Alternatively, use a training loop like so, this will feature engineer + run experiments for no retrain & always retrain for all replayed data in /incremental :
`./r s/processing experiment_loop --input runs/exp/tencent/1063/1m/iops/replayed/incremental/ --output runs/exp/tencent/1063/1m/iops/`

6. Given folder path, we can train a model like so. This trains a non-retraining model using the first chunk.
`./r s/train initial_only --data ./runs/exp/tencent/1063/1m/iops/processed/gradual/105_117/ -o ./runs/exp/tencent/1063/1m/iops/experiments/gradual/105_117/`

7. We can plot the results of the experiment 1 by 1 like so
`./r s/train plot_exp -i output/1063/iops/experiments/gradual/6350_6650 -o output/1063/iops/experiments/gradual/6350_6650/plot/`

or, run a plotting loop by giving the folder path of your experiments. The output will be in the /plot subdirectory within each experiment folder
`./r s/train plot_exp_glob -i output/1063/iops/experiments`
`./r s/train plot_exp_glob -i ./runs/exp/tencent/1063/1m/iops/experiments`

Misc:
- [Plotting concated cdf] `./r cdf_concat_from_replay_dir_glob -d /home/cc/clio/output/iops/replayed/ -o ./plot_cdf -f`
- [Rescale] `./r s/processing.sh rescale_data --input "./runs/raw/tencent/split/1063" --output "./output/iops/rescaled/1063" --metric iops --multiplier 1.2`
`./r s/processing.sh rescale_data --input "./runs/raw/tencent/split/1063" --output "./output/iops/rescaled/1063/IOPS/0.5" --metric iops --multiplier 0.5`
`./r s/processing.sh rescale_data --input "./runs/raw/tencent/split/1063" --output "./output/iops/rescaled/1063/IOPS/1.5" --metric iops --multiplier 1.5`

Rsync: `rsync -Pavrz runs/exp clio-box:/home/runs`

# Rayst
rsync -Pavrz 192.5.87.59:/home/cc/clio/output/1063/iops/experiments/incremental runs/exp/tencent/1063/1m/iops/experiments/

# Raystor
rsync -Pavrz 192.5.87.101:/home/cc/clio/output/1063/iops/experiments/incremental runs/exp/tencent/1063/1m/iops/experiments/

./r cdf_concat_from_replay_dir_glob -d runs/exp/tencent/1063/1m/iops/replayed -o runs/exp/tencent/1063/1m/iops/plot_cdf_100 -f --max 0.99

### Analysis

#### Tencent

See `s/raw/tencent.sh`

Download `tencent` raw data, ask William Nixon or Ray or just go directly to [SNIA](http://iotta.snia.org/traces/parallel/27917). This data is huge, so we ended up having our own processing code to help you analyzing this data.

- Count volume map: `./r s/raw/tencent count_volume_map --input <input directory that contains *.tgz> --output <output directory>`
- Count volume reduce: `./r s/raw/tencent count_volume_reduce --input <input directory from count volume map result> --output <output directory>`
- Pick volume: `./r s/raw/tencent pick_volume --input <input directory that contains *.tgz> --output <output directory> --volume <chosen volume>`
- Split (will split based on provided window and convert to replayer format): `./r s/raw/tencent split --input <input directory that contains *.tgz (preferably from pick volume)> --output <output directory --window <window>`
- Calculate characteristic (From splitted file): `./r s/processing calc_raw_characteristic --input <input dir that contains *.tgz from split>  --output <> --window 1m`
- Calculate characteristic from replayed file (replayed format): `/r s/processing calc_replayed_characteristic --input <Glob to *.csv> --output <> --window 1`

#### ALIBABA
We use duckDB to filter data from Alibaba. A duckDB database is uploaded to chameleon. The overview of the steps is as follows:
Note: You need at least ~300 GB of storage to be safe, alibaba size along is 200 gb. Get a storage node to be safe.
1. Download/setup duckDB in your machine.
Please use one of the following binaries:
https://github.com/duckdb/duckdb/releases/download/v1.0.0/duckdb_cli-linux-amd64.zip
https://github.com/duckdb/duckdb/releases/download/v1.0.0/duckdb_cli-linux-aarch64.zip

Usage example
./duckdb

`wget https://github.com/duckdb/duckdb/releases/download/v1.0.0/duckdb_cli-linux-amd64.zip
 unzip duckdb_cli-linux-amd64.zip
 chmod +x duckdb
 ./duckdb
`
2. Download the duckDB database containing alibaba, through the following command `swift --os-auth-type v3applicationcredential --os-application-credential-id X --os-application-credential-secret X download clio-data -p iotrace_master`
You might have to change the column names
`
ALTER TABLE alibaba_whole RENAME COLUMN column0 TO device_id;
ALTER TABLE alibaba_whole RENAME COLUMN column1 TO opcode;
ALTER TABLE alibaba_whole RENAME COLUMN column2 TO 'offset';
ALTER TABLE alibaba_whole RENAME COLUMN column3 TO length;
ALTER TABLE alibaba_whole RENAME COLUMN column4 TO timestamp;
`
2. Read the database into your machine (`.open iotrace_master`), and query for a specific device ID. Do the following query
` COPY ( SELECT * FROM alibaba_whole WHERE device_id = 'X' ) TO './device_X_csv_glob' (FORMAT CSV, per_thread_output True); `
See this link for list of device characteristics `https://docs.google.com/spreadsheets/d/1i47Uc4lu7pkJ-33Oa_xW9JHyYG8E_8xq4E6UOTiQ56I/edit?gid=1181123042#gid=1181123042`
This will output a series of .csv files in an output directory, ready to be splitted. A successful execution will give you a folder with a lot of csv files. This split is necessary to run parallel processing on these files.
3. Run the alibaba pipeline to split, the command is as follows
` ./r s/raw/alibaba split --input <path to filtered csv folder> --output ./runs/raw/alibaba/split/X/1m/`
A successful execution will give you a splitted output file, in .tgz format. This format of file is ready to be replayed.
4. To calculate characteristic from the above split file, run this command
`./r s/raw/alibaba calc_characteristics --input <path to splitted device folder> --output ./runs/raw/alibaba/characteristic/X/1m/ --window 1m`
A succesful execution will give you a characteristic.csv file.
5. Characteristic from replayed file is the same with tencent `/r s/processing calc_replayed_characteristic --input <Glob to *.csv> --output <> --window 1`

With /split and /characteristic, alibaba data is now preprocessed and ready for the usual pipeline.
