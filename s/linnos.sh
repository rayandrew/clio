
# ./r labeling_feat_linnos --data-dir runs/exp/tencent/1063/1m/iops/replayed/ --output ./runs/exp/tencent/1063/1m/iops/linnos
labeling_feat_linnos() {
  # echo "HE"
  local data_dir output
  data_dir=$(parse_opt_req "data-dir:d" "$@")
  output=$(parse_opt_req "output:o" "$@")
  # echo "CALLED"

  for f in $(find "$data_dir" -type f -name "*.csv"); do
    drift_type=$(echo "$f" | grep -oP '(?<=replayed/)(gradual|incremental|sudden|recurring)')
    start_end=$(echo "$f" | grep -oP '\d+_\d+')

    if [[ -f "$output/labeling/$drift_type/$start_end/done" ]]; then
      echo "Already processed, skipping"
      continue
    fi

    echo "Processing labeling.py -data $f -output $output/labeling/$drift_type/$start_end"
    python labeling.py -file $f -output_dir $output/labeling/$drift_type/$start_end
  done

  for f in $(find "$data_dir" -type f -name "*.csv"); do
    drift_type=$(echo "$f" | grep -oP '(?<=replayed/)(gradual|incremental|sudden|recurring)')
    start_end=$(echo "$f" | grep -oP '\d+_\d+')
    # make done file
    if [[ -f "$output/labeling/$drift_type/$start_end/done" ]]; then
      echo "Already processed, skipping"
      continue
    fi
    touch "$output/labeling/$drift_type/$start_end/done"
  done

  for f in $(find "$output/labeling" -type f -name "done"); do
    echo "Processing $f"
    drift_type=$(echo "$f" | grep -oP '(gradual|incremental|sudden|recurring)')
    start_end=$(echo "$f" | grep -oP '\d+_\d+')

    if [[ -f "$output/feat/$drift_type/$start_end/done" ]]; then
      echo "Already processed, skipping"
      continue
    fi

    echo "feature engineering -data_dir $f -output $output/feat/$drift_type/$start_end"
    python linnos/feat.py -dir_input $f -output $output/feat/$drift_type/$start_end

    # touch done file, mkdir
    mkdir -p "$output/feat/$drift_type/$start_end"
    touch "$output/feat/$drift_type/$start_end/done"
  done
}

# ./r run_model_linnos --data-dir runs/exp/tencent/1063/1m/iops/linnos/feat --output ./runs/exp/tencent/1063/1m/iops/linnos/
run_model_linnos() {
  local data_dir output
  data_dir=$(parse_opt_req "data-dir:d" "$@")
  output=$(parse_opt_req "output:o" "$@")

  for f in $(find "$data_dir" -type f -name "done"); do
    drift_type=$(echo "$f" | grep -oP '(gradual|incremental|sudden|recurring)')
    start_end=$(echo "$f" | grep -oP '\d+_\d+')

    if [[ -f "$output/exp/$drift_type/$start_end/no_retrain/results.csv" ]]; then
      echo "Already processed, skipping"
      continue
    fi

    # get dir name of f
    f=$(dirname "$f")

    echo "Running exp -datadir $f -output $output/exp/$drift_type/$start_end"
    python linnos/model.py -train_eval_split 70_30 -dataset_dir $f -output $output/exp/$drift_type/$start_end/no_retrain

    # # touch done file
    # touch "$output/exp/$drift_type/$start_end/done"
  done
}
# ./r run_model_linnos --data-dir runs/exp/tencent/1063/1m/iops/linnos/feat --output ./runs/exp/tencent/1063/1m/iops/linnos/
run_model_linnos_retrain() {
  local data_dir output
  data_dir=$(parse_opt_req "data-dir:d" "$@")
  output=$(parse_opt_req "output:o" "$@")

  for f in $(find "$data_dir" -type f -name "done"); do
    drift_type=$(echo "$f" | grep -oP '(gradual|incremental|sudden|recurring)')
    start_end=$(echo "$f" | grep -oP '\d+_\d+')

    if [[ -f "$output/exp/$drift_type/$start_end/retrain/results.csv" ]]; then
      echo "Already processed, skipping"
      continue
    fi

    # get dir name of f
    f=$(dirname "$f")

    echo "Running exp -datadir $f -output $output/exp/$drift_type/$start_end"
    python linnos/model.py -train_eval_split 70_30 -dataset_dir $f -output $output/exp/$drift_type/$start_end/retrain -retrain true

    # touch done file
    # touch "$output/exp/$drift_type/$start_end/done"
  done
}