from workflow.helper import get_flashnet_data

NUM_CHUNK = 12

def get_flashnet_60_mins_chunks(storage, trace):
  return [get_flashnet_data(storage, f"trace_ori_60mins/{trace}/nvme0n1/splitted/min_5.0/profile/chunk_{i}/profile_v1.feat_v6_ts.readonly.dataset") for i in range(NUM_CHUNK)]
  # return [get_flashnet_data(storage, f"trace_ori_60mins/{trace}/nvme0n1/splitted/min_5.0/profile/chunk_{i}/profile_v1.labeled") for i in range(NUM_CHUNK)]

rule flashnet_clustering:
  input:
    lambda wildcards: get_flashnet_60_mins_chunks(storage, wildcards.trace),
  output:
    "data/analysis/flashnet/{trace}/clustering/stats.stats",
  shell:
    "python -m clio.characteristics.clustering $(dirname {input[0]}) --output $(dirname {output[0]})"
