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
sudo apt-get install gnuplot-qt ghostscript
```


python -m clio.flashnet.cli.characteristic concept_finder ./test "/home/cc/clio/runs/raw/tencent/characteristic/1063/1m/characteristic.csv"

./s/femu.sh replay_list --range-list "/home/cc/clio/test/iops/picked_drifts.csv" --data-dir "/home/cc/clio/runs/raw/tencent/split/1063" --output "./data/test"

./s/femu.sh postprocess --input /home/cc/clio/data/test/gradual/1_2/raw --output /home/cc/clio/data/test/gradual/1_2/