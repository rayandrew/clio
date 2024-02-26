###################################################
# [Global variables]
# Change if you need to but don't commit changes
###################################################

ENV_NAME := clio
PATH := $(MAMBA_ROOT_PREFIX)/envs/$(ENV_NAME)/bin:$(PATH)
PYTHONPATH := $(CURDIR):$(PYTHONPATH)
CUBLAS_WORKSPACE_CONFIG := ":4096:8"
TF_CPP_MIN_LOG_LEVEL := "3"

export ENV_NAME
export PATH
export PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG
export TF_CPP_MIN_LOG_LEVEL

###################################################
# [Targets]
###################################################

.PHONY:

python:
	python

which-python:
	which python

###################################################
# [1] MSRC
###################################################

msrc: \
	data/analysis/msrc/hm/done
	# data/standardized/msrc/mds/done \
	# data/standardized/msrc/proj/done \
	# data/standardized/msrc/rsrch/done \
	# data/standardized/msrc/src1/done \
	# data/standardized/msrc/src2/done \
	# data/standardized/msrc/stg/done \
	# data/standardized/msrc/ts/done \
	# data/standardized/msrc/usr/done \
	# data/standardized/msrc/wdev/done \
	# data/standardized/msrc/web/done
	# data/standardized/msrc/prxy/done \
		#

data/standardized/msrc/%/%.trace: raw-data/msrc/%_*.csv.gz
	@echo "Processing MSRC $(notdir $@)"
	python -m clio.trace.standardizer msrc $^ --output $(dir $@)

# HACK: This is a workaround to avoid reprocessing the same data
data/standardized/msrc/%/done: data/standardized/msrc/%/%.trace
	touch $@

data/analysis/msrc/%/done: data/standardized/msrc/%/done
	@echo "Analyzing MSRC $(notdir $@)"
	python -m clio.trace.analyzer full $(dir $<)/$*.trace --output $(dir $@) --query "disk_id == '1'"

msrc-quick-analyze-%: data/standardized/msrc/%/done
	python -m clio.trace.analyzer quick data/standardized/msrc/$*/$*.trace

msrc-clean-analysis:
	rm -rf data/analysis/msrc
