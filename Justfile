###################################################
# [Global variables]
# Change if you need to but don't commit changes
###################################################

export ENV_NAME := "flashnet"
export PATH := env_var("MAMBA_ROOT_PREFIX") + "/envs/" + ENV_NAME + "/bin:" + env_var("PATH")
export PYTHONPATH := justfile_directory() + ":" + env_var_or_default("PYTHONPATH", "")
export CUBLAS_WORKSPACE_CONFIG := ":4096:8"
export TF_CPP_MIN_LOG_LEVEL := "3"

###################################################
# [Commands]
###################################################

msr-hm +args="":
  #!/usr/bin/env make -f

  .PHONY: all

  data/msr/hm/log.txt: raw-data/msr/hm_*.csv.gz
  	@echo "Running MSR-HM"
  	@echo "aa"
