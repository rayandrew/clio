import os

from pathlib import Path
from workflow.helper import get_flashnet_data
from clio.utils.trace_pd import trace_get_dataset_paths

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

NUM_CHUNK = 12

def get_flashnet_60_mins_chunks(storage, trace):
  return [get_flashnet_data(storage, f"trace_ori_60mins/{trace}/nvme0n1/splitted/min_5.0/profile/chunk_{i}/profile_v1.feat_v6_ts.readonly.dataset") for i in range(NUM_CHUNK)]
  # return [get_flashnet_data(storage, f"trace_ori_60mins/{trace}/nvme0n1/splitted/min_5.0/profile/chunk_{i}/profile_v1.labeled") for i in range(NUM_CHUNK)]

def dir_of_chunk(path: str | Path):
  path = Path(path)
  return path.parent.parent.parent

rule flashnet_clustering:
  input:
    lambda wildcards: get_flashnet_60_mins_chunks(storage, wildcards.trace),
  output:
    "data/analysis/flashnet/{algo}/{trace}/clustering/stats.stats",
  shell:
    "python -m clio.characteristics.clustering {wildcards.algo} $(dirname $(dirname {input[0]})) --output $(dirname {output[0]})"

def get_generated_window(data_path: str | Path, initial_data_path: str | Path | None = None):
  data_path = Path(data_path)
  data_paths = trace_get_dataset_paths(data_path) # return dict of paths
  if initial_data_path is not None:
    initial_data_path = Path(initial_data_path)
    initial_data_paths= trace_get_dataset_paths(initial_data_path)
    return initial_data_paths + data_paths
  return data_paths

rule flashnet_workload_prediction:
  input:
    # get_generated_window(data_path="raw-data/flashnet/generated-window/test_8_hours", initial_data_path="raw-data/flashnet/generated-window/train_5_min"),
    get_generated_window("raw-data/flashnet/generated-window/test_8_hours"),
  output:
    # "data/analysis/flashnet/workload-prediction.window_1min.duration_1h/stats.stats",
    # "data/analysis/flashnet/workload-prediction.window_1min.duration_1h/log.txt",
    # "data/analysis/flashnet/workload-prediction.window_1min.duration_10min/stats.stats",
    # "data/analysis/flashnet/workload-prediction.window_1min.duration_10min/log.txt",
    "data/analysis/flashnet/workload-prediction.window_{window}.duration_{duration}/stats.stats",
    "data/analysis/flashnet/workload-prediction.window_{window}.duration_{duration}/results.csv",
    "data/analysis/flashnet/workload-prediction.window_{window}.duration_{duration}/models.csv",
    # directory("data/analysis/flashnet/workload-prediction.window_{window}.duration_{duration}/models"),
    directory("data/analysis/flashnet/workload-prediction.window_{window}.duration_{duration}/window"),
    "data/analysis/flashnet/workload-prediction.window_{window}.duration_{duration}/log.txt",
  shell:
    """
    python -m clio.flashnet.exp.workload_prediction \
      "raw-data/flashnet/generated-window/test_8_hours" \
      --output $(dirname {output[0]}) \
      --window-size {wildcards.window} \
      --duration {wildcards.duration} \
      --prediction-batch-size 16384
      
    # --duration 5
    # --initial-data-dir "raw-data/flashnet/generated-window/train_5_min"
    """

rule flashnet_workload_prediction_analyze:
  input:
    "data/analysis/flashnet/workload-prediction.window_{window}.duration_{duration}/results.csv",
    "data/analysis/flashnet/workload-prediction.window_{window}.duration_{duration}/models.csv",
    "data/analysis/flashnet/workload-prediction.window_{window}.duration_{duration}/window",
  output:
    directory("data/plot/flashnet/workload-prediction.window_{window}.duration_{duration}"),
  shell:
    """
    python -m clio.flashnet.exp.workload_prediction_analyze $(dirname {input[0]}) \
      --output {output[0]}
    """
  