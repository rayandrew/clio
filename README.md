# CLIO

## Getting started

### Dependencies

- [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (Conda also works)
- Latex (for generating PDFs and combining plots)

### Installation

```bash
micromamba env create --name clio --file env.yaml
# upgrade keras because tensorflow shipped their own version
pip install --upgrade keras
```

### Activation

```bash
micromamba activate clio
```

### Creating Env File (only one time)

```bash
touch .env
```

Add the following line to the `.env` file and ask Ray for the information:

```bash
SNAKEMAKE_STORAGE_SFTP_USERNAME=<username>
SNAKEMAKE_STORAGE_SFTP_PASSWORD=<password>
```

## Processing

### MSRC

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
