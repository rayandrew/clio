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
pip install --upgrade fitsne
```

### Activation

```bash
micromamba activate clio
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

## Reports

### March -- Week 5

#### Sent Date

March 29, 2024

#### Documents

Google Docs: [LINK](https://docs.google.com/document/d/10tkwZtRON6IN7gGXYpx5x-jqCDNy61rAJDvcmvhpqzM/edit)

#### Description
x
- We try to compare simple data sampling mechanisms that answers our research question: **how to reduce overhead in retraining data due to the scale of I/O data, e.g. in 1 min we can have more than 1M data?**
- We use prediction probability-based sampling mechanism that does not need to have "ground truth" and can be collected during prediction

#### Reproduce

I am using `var_0`, `var_1` and `var_3` as `FLASHNET_EXP_NAME`

```bash
./makesure weekly_report.march.year_24.week_5 -D FLASHNET_EXP_NAME=<FLASHNET_EXP_NAME> -D FLASHNET_EXP_DIR_SUFFIX="" -D EPOCHS=20 -D CUDA=<CUDA_DEVICE>
```

#### Conclusions

- Probability-based sampling shows improvement in train computational efficiency and also model performance compare to retraining using all data

--------------------------------------------------

### April -- Week 1

#### Sent Date

April 3, 2024

#### Documents

Google Docs: [LINK](https://docs.google.com/document/d/1qblSFob4Rs2hyM-suG5MkEcpzcce_rNsSMYhfYyULsg/edit)

#### Description

Here, we try to combine the models' performance between single model and description models.
For prediction, we use average for all models prediction probabilities

#### Reproduce

I am using `var_0`, `var_1` and `var_3` as `FLASHNET_EXP_NAME`

```bash
./makesure weekly_report.april.year_24.week_1 -D FLASHNET_EXP_NAME=<FLASHNET_EXP_NAME> -D FLASHNET_EXP_DIR_SUFFIX="" -D EPOCHS=20 -D CUDA=<CUDA_DEVICE>
```

#### Conclusions:

- Multiple models beat the single models
- "Worse" training data can still be used in the multiple models

--------------------------------------------------