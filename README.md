# CLIO

## Getting started

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

### Documents

- [Trace Editor](./docs/trace-editor.md)


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
swift --os-auth-type v3applicationcredential --os-application-credential-id da8eb9b3943c452fa4183fad9d16e58c --os-application-credential-secret AUp1cJZ9ZHiUnAaPuXE8V55NFZ3Cu2Us4DQzXN0wQZIAvFQJ0cDURAy7NLzjckwfefAQsSDbiFU92JvN0cfs0A download clio-data -p runs
```

This will download:
- Characteristic, a csv containing metrics like IOPS, IAT, etc with windows split every 1m, 10m, 1h, etc. 
- Split, I/O traces split into 1 minute intervals

2. We will be using the characteristic file to get an idea of where drift might be, using a target metric. We can run a script that will produce a list of drifts like so, in this command, using the IOPS characteristic of device 1063 to find drift in 1 minute windows: 
``` ./r s/processing.sh compile_and_get_drifts -o ./output -i ./runs/raw/tencent/characteristic/1063/1m/characteristic.csv -m iops ```

3. You can also plot the drifts from the previous step by the following command
``` ./r s/processing.sh plot_drifts -p ./output/iops/data/ -o ./output/iops/replay_plot/ ```

3. The csv will give a lot of potential drifts. We need to select a subset to replay. Set the column in to_be_picked_drifts.csv to 'y' if you want that drift replayed. Note that using 1 minute windows, replaying will take approximately ~1m too. So replaying windows from idx 100-200 will take ~100 minutes.

4. Replay the chunks marked by 'y' in the csv file by running. Range-list is a csv to read. This will loop through the CSV file, get the rows marked by 'y', then replay chunks from start to finish in FEMU (an SSD emulator). Data_dir should point to the folder containing files from chunks_0 to chunks_XXX.
``` ./r s/femu.sh replay_list --range-list "output/iops/selected_drifts.csv" --data-dir "./runs/raw/tencent/split/1063" --output "./output/iops/replayed" --time-split 1m ```

5. Once done replaying, we can label and feature engineer everything in the replayed folder. This will output files that will be used to train our models.
``` ./r s/processing.sh postprocess --input ./output/iops/replayed/gradual/105_117/raw --output ./output/iops/processed/gradual/105_117/ ```

6. Given folder path, we can train a model like so. This trains a non-retraining model using chunk_0.
``` ./r s/train.sh initial_only --data ./output/iops/processed/gradual/105_117/ -o ./output/iops/experiments/gradual/105_117/```
7. We can plot the results of the experiment like so
 ``` ./r s/train.sh plot_exp -i ./output/iops/experiments/gradual/105_117/ -o ./output/iops/experiments/gradual/105_117/plot/ ```


Misc:
- [Plotting concated cdf] ./r cdf_concat_from_replay_dir_glob -d /home/cc/clio/output/iops/replayed/ -o ./plot_cdf 