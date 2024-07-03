import yaml
import argparse
import pandas as pd
import numpy as np
import os, datetime

from alibi_detect.cd import KSDrift, CVMDrift

def process_output(args, df_preds):
    
    if args.output_mode == 'all':
        results = df_preds[['Window_Start', 'Window_End', 'Is_Drift']]

    elif args.output_mode == 'rank':
        
        # Rank the p-values from lowest to highest
        df_sorted = df_preds.sort_values(by=args.rank_by).loc[df_preds['Is_drift'] == 1]

        # Select the top five p-values and their corresponding window indices
        cutoff = int(args.pecentage * len(df_sorted))
        
    
    return results
    

def process_preds(step_size, window_len, preds):
    
    # Initialize lists to store data for plotting
    window_indices = []
    is_drift_values = []
    p_val_values = []
    distance_values = []
    threshold_values = []
    window_start_index = []
    window_end_index = []

    # Extract data for each prediction in preds
    for idx, pred in enumerate(preds):
        window_indices.append(idx + 1)  # Assuming window index starts from 1
        window_start_index.append((idx + 1)*step_size)
        window_end_index.append((idx + 1)*step_size + window_len)
        is_drift_values.append(pred['data']['is_drift'][0])
        p_val_values.append(pred['data']['p_val'][0])
        distance_values.append(pred['data']['distance'][0])
        threshold_values.append(pred['data']['threshold'])

    # Create a DataFrame from the lists
    data_dict = {
        'Window_Index': window_indices,
        'Window_Start': window_start_index,
        'Window_End': window_end_index,
        'Is_Drift': is_drift_values,
        'P_Value': p_val_values,
        'Distance': distance_values,
        'Threshold': threshold_values
    }

    df_preds = pd.DataFrame(data_dict)
    return df_preds

def moving_prediction(args):
    
    df_data = pd.read_csv(args.data_path)
    
    window_len = args.window_len
    step_size = args.step_size
    category = args.feature_name
    ref_start = 0
    ref_end = window_len
    start = window_len
    end = start + window_len
    preds = []
    
    while end < len(df_data):
        data_ref = df_data[category].iloc[ref_start:ref_end].to_numpy()
        if args.ks_drift:
            cd = KSDrift(data_ref, p_val=args.threshold)
        elif args.cvm_drift:
            cd = CVMDrift(data_ref, p_val=args.threshold)

        x = df_data[category].iloc[start:end].to_numpy()
        pred = cd.predict(x, drift_type='feature', return_p_val=True, return_distance=True)
        preds.append(pred)
        
        ref_start += step_size
        ref_end = ref_start + window_len
        start += step_size
        end = start + window_len
        
    df_preds = process_preds(step_size, window_len, preds)
    
    return df_preds
    
    
def fixed_prediction(args):
    
    df_data = pd.read_csv(args.data_path)
    
    window_len = args.window_len
    step_size = args.step_size
    category = args.feature_name
    start = window_len
    end = start + window_len
    preds = []
    
    data_ref = df_data[category].head(window_len).to_numpy()
    if args.ks_drift:
        cd = KSDrift(data_ref, p_val=args.threshold)
    elif args.cvm_drift:
        cd = CVMDrift(data_ref, p_val=args.threshold)
        
    while end < len(df_data):
        x = df_data[category].iloc[start:end].to_numpy()
        pred = cd.predict(x, drift_type='feature', return_p_val=True, return_distance=True)
        preds.append(pred)
        
        start += step_size
        end = start + window_len
    
    df_preds = process_preds(step_size, window_len, preds)
    
    return df_preds
    

def main(args):
    
    output_path = os.path.join(args.output_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(output_path, exist_ok=True)
    
    if args.ref_mode == 'fixed':
        if args.ref_start == None or args.ref_end == None:
            raise ValueError("Please provide a reference period start and end date")
        
        df_preds = fixed_prediction(args)

    else:
        df_preds = moving_prediction(args)
        
    
    # results = process_output(args, df_preds)
    
    df_preds.to_csv(os.path.join(output_path, 'drift_predictions.csv'), index=False)
    # results.to_csv(os.path.join(output_path, 'selected_drifts.csv'), index=False)
        
    

if __name__ == "__main__":
    # Load YAML configuration
    parser = argparse.ArgumentParser()
    
    with open('./config/config.yaml', 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    for key, value in config_data.items():
        parser.add_argument(f'--{key}', type=str, default=value, help=f'{key} argument from YAML')
    args = parser.parse_args()
    
    main(args)