# CLIO

## Data

### MSRC

Download the MSRC dataset from [here](http://iotta.snia.org/traces/block-io/388). Extract and put everything to `raw-data/msrc`.

- Quick analysis on the MSRC dataset:

```bash
# snakemake data/analysis/msrc/<trace>/quick/stats.stats --cores 1 # --force --rerun-incomplete

snakemake data/analysis/msrc/hm/quick/stats.stats --cores 1 # --force --rerun-incomplete
```

- Window analysis on the MSRC dataset:

```bash
# snakemake data/analysis/msrc/<trace>/win_<second>s/stats.stats --cores 1 # --rerun-incomplete --force
snakemake data/analysis/msrc/hm/win_60s/stats.stats --cores 1 # --rerun-incomplete --force
```
