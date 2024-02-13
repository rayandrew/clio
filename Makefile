###################################################
# [Global variables]
# Change if you need to but don't commit changes
###################################################

ENV_NAME := flashnet
PATH := $(MAMBA_ROOT_PREFIX)/envs/$(ENV_NAME)/bin:$(PATH)
PYTHONPATH := $(CURDIR):$(PYTHONPATH)
CUBLAS_WORKSPACE_CONFIG := ":4096:8"
TF_CPP_MIN_LOG_LEVEL := "3"

export PATH
export PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG
export TF_CPP_MIN_LOG_LEVEL

###################################################
# [Targets]
###################################################

.PHONY:

###################################################
# [1] MSRC
###################################################

# data/msr/hm/done: raw-data/msr/hm_*.csv.gz
# 	@echo "Running MSR-HM"
# 	python -m clio.preprocess.msr msrc $^ --output $(dir $@)
# 	touch $@

msrc: data/msrc/hm/done \
	data/msrc/mds/done \
	data/msrc/proj/done \
	data/msrc/prxy/done \
	data/msrc/rsrch/done \
	data/msrc/src1/done \
	data/msrc/src2/done \
	data/msrc/stg/done \
	data/msrc/ts/done \
	data/msrc/usr/done \
	data/msrc/wdev/done \
	data/msrc/web/done

data/msrc/%/done: raw-data/msrc/%_*.csv.gz
	@echo "Running MSRC $(notdir $@)"
	python -m clio.preprocess.msr msrc $^ --output $(dir $@)
	touch $@
