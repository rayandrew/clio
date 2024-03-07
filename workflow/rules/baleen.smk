## Note this is for SINGLE TRACE ONLY. Please use the wildcards accordingly. ##
## Refer to list of files you can get here https://ftp.pdl.cmu.edu/pub/datasets/Baleen24/storage/
## Year 201910, 202110,20230325
## Region Region1-6, depend on year
## Trace full_0_1.trace
## ex: snakemake data/standardized/tectonic/201910/Region1/full_0_1.trace -c 1 --rerun-incomplete
rule baleen_single_trace:
  input:
    lambda wildcards: http(storage, "https://ftp.pdl.cmu.edu/pub/datasets/Baleen24/storage/{wildcards.year}/{wildcards.region}/{wildcards.trace}.trace"),
  output:
    "data/standardized/tectonic/{year}/{region}/{trace}.trace",
  shell:
    "python -m clio.trace.standardizer tectonic {input} --output {output[0]}"
