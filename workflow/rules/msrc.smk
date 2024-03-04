from workflow.helper import get_clio_msrc_data

rule msrc_standardize:
  input:
    lambda wildcards: get_clio_msrc_data(storage, wildcards.trace),
  output:
    "data/standardized/msrc/{trace}/{trace}.trace",
    "data/standardized/msrc/{trace}/log.txt",
  log:
    "logs/standardized/msrc/{trace}/{trace}.log",
  shell:
    "python -m clio.trace.standardizer msrc {input} --output $(dirname {output[0]})"

rule msrc_whole_analyze:
  input:
    "data/standardized/msrc/{trace}/{trace}.trace",
  output:
    "data/analysis/msrc/{trace}/whole/stats/stats.stats",
    "data/analysis/msrc/{trace}/whole/stats/stats.msgpack",
    "data/analysis/msrc/{trace}/whole/stats/log.txt",
  log:
    "logs/analysis/msrc/{trace}/whole/{trace}.log",
  shell:
    "python -m clio.trace.analyzer whole {input} --output $(dirname {output[0]})"

rule msrc_whole_plot_characteristic:
  input:
    "data/analysis/msrc/{trace}/whole/stats/stats.msgpack",
  output:
    directory("data/analysis/msrc/{trace}/whole/plots"),
    "data/analysis/msrc/{trace}/whole/plots/log.txt",
  log:
    "logs/analysis/msrc/{trace}/whole/plots/{trace}.log",
  shell:
    "python -m clio.trace.plotter characteristic {input} --output $(dirname {output[0]})"

rule msrc_window_analyze:
  input:
    "data/standardized/msrc/{trace}/{trace}.trace",
  output:
    "data/analysis/msrc/{trace}/win_{seconds}s/stats/stats.stats",
    "data/analysis/msrc/{trace}/win_{seconds}s/stats/stats.msgpack",
    "data/analysis/msrc/{trace}/win_{seconds}s/stats/log.txt",
  log:
    "logs/analysis/msrc/{trace}/win_{seconds}s/{trace}.log",
  shell:
    """
    python -m clio.trace.analyzer window {input} --output $(dirname {output[0]}) --window {wildcards.seconds} --log-level INFO --query "data['ts_record'] >= 0"
    """

rule msrc_window_plot_characteristics:
  input:
    "data/analysis/msrc/{trace}/win_{seconds}s/stats/stats.msgpack",
  output:
    directory("data/analysis/msrc/{trace}/win_{seconds}s/plots"),
    "data/analysis/msrc/{trace}/win_{seconds}s/plots/log.txt",
  log:
    "logs/analysis/msrc/{trace}/win_{seconds}s/plots/{trace}.log",
  shell:
    """
    python -m clio.trace.plotter characteristics {input} --output {output[0]} --log-level INFO --query "start_ts >= 0"
    """

rule msrc_window_pdf_characteristics:
  input:
    "data/analysis/msrc/{trace}/win_{seconds}s/plots",
  output:
    "data/analysis/msrc/{trace}/win_{seconds}s/pdfs/characteristics.pdf",
    "data/analysis/msrc/{trace}/win_{seconds}s/pdfs/log.txt",
  log:
    "logs/analysis/msrc/{trace}/win_{seconds}s/pdfs/{trace}.log",
  shell:
    """
    python -m clio.trace.pdfs characteristics {input} --output {output[0]} --log-level INFO --query "start_ts >= 0"
    """
