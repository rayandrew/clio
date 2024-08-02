import yaml
import argparse
import pandas as pd
import numpy as np
import os, datetime
import matplotlib.pyplot as plt

from alibi_detect.cd import KSDrift, CVMDrift
    

def process_preds(feature, ks_preds, cvm_preds):
    
    ks_rows = [(record['data']['distance'][0], record['data']['p_val'][0]) for record in ks_preds]
    ks_preds = pd.DataFrame(ks_rows, columns=[f'{feature}_ks_distance', f'{feature}_ks_p_val'])
    
    ks_preds[f'{feature}_ks_label'] = 0
    sorted_df = ks_preds.sort_values(by=f'{feature}_ks_p_val')
    lowest_ten_indices = sorted_df.head(10).index
    highest_ten_indices = sorted_df.tail(10).index
    ks_preds.loc[lowest_ten_indices, f'{feature}_ks_label'] = 1
    ks_preds.loc[highest_ten_indices, f'{feature}_ks_label'] = 2
    
    cvm_rows = [(record['data']['distance'][0], record['data']['p_val'][0]) for record in cvm_preds]
    cvm_preds = pd.DataFrame(cvm_rows, columns=[f'{feature}_cvm_distance', f'{feature}_cvm_p_val'])
    sorted_cvm = cvm_preds.sort_values(by=f'{feature}_cvm_p_val')
    lowest_ten_cvm_indices = sorted_cvm.head(10).index
    highest_ten_cvm_indices = sorted_cvm.tail(10).index
    cvm_preds[f'{feature}_cvm_label'] = 0
    cvm_preds.loc[lowest_ten_cvm_indices, f'{feature}_cvm_label'] = 1
    cvm_preds.loc[highest_ten_cvm_indices, f'{feature}_cvm_label'] = 2
    
    # Counting the number of labels for KS
    ks_label_counts = ks_preds[f'{feature}_ks_label'].value_counts()
    print(f"KS Labels for {feature}:")
    print(ks_label_counts)
    
    # Counting the number of labels for CvM
    cvm_label_counts = cvm_preds[f'{feature}_cvm_label'].value_counts()
    print(f"CvM Labels for {feature}:")
    print(cvm_label_counts)
    
    df_preds = pd.concat([ks_preds, cvm_preds], axis = 1)
    
    return df_preds

def moving_prediction(args, feature):
    
    df_data = pd.read_csv(args.data_path)
    
    window_len = args.window_len
    step_size = args.step_size
    ref_start, ref_end = 0, window_len
    start, end = window_len, window_len + window_len
    ks_preds, cvm_preds = [], []
    
    while end < len(df_data):
        data_ref = df_data[feature].iloc[ref_start:ref_end].to_numpy()
        x = df_data[feature].iloc[start:end].to_numpy()
        
        KS_cd = KSDrift(data_ref, p_val=args.threshold)
        CVM_cd = CVMDrift(data_ref, p_val=args.threshold)
        
        ks_pred = KS_cd.predict(x, drift_type='feature', return_p_val=True, return_distance=True)
        ks_preds.append(ks_pred)
        
        cvm_pred = CVM_cd.predict(x, drift_type='feature', return_p_val=True, return_distance=True)
        cvm_preds.append(cvm_pred)
        
        ref_start += step_size
        ref_end = ref_start + window_len
        start += step_size
        end = start + window_len
        
    df_preds = process_preds(feature, ks_preds, cvm_preds)
    
    return df_preds
    
def save_plots(args, output_path, df_preds):
    
    df_data = pd.read_csv(args.data_path)
    
    if df_preds.columns.duplicated().any():
        raise ValueError("Duplicate column names found in df_preds.")
    
    for feature in args.features:
        
        for drift_detector in ['ks', 'cvm']:
        
            label_column = f'{feature}_{drift_detector}_label'
            
            for label in [1, 2]:
                label_df = df_preds[df_preds[label_column] == label]

                fig, axs = plt.subplots(5, 2, figsize=(15, 20))
                fig.suptitle(f'{feature} - Label {label}')
                axs = axs.ravel()  # Flatten the 2D array of axes

                for i, (start, end) in enumerate(zip(label_df['start'], label_df['end'])):
                    if i >= 10:
                        break  # Limit to 10 subplots
                    
                    # Slicing df_data based on start and end indices
                    sliced_data = df_data.iloc[start:end]
                    feature_data = sliced_data[feature]

                    # Plotting the feature data range
                    axs[i].plot(range(start, end), feature_data, label=f'{feature} data')
                    axs[i].set_title(f'Start: {start}, End: {end}')
                    axs[i].legend()

                plt.tight_layout()
                plt.subplots_adjust(top=0.95)
                
                # Save the figure
                save_path = f"{output_path}/{feature}_label_{label}_{drift_detector}.png"
                plt.savefig(save_path)
                plt.close()  # Close the figure after saving to free up memory

                print(f"Plot saved to {save_path}")
    
    return
    

def main(args):
    
    filename = f'{args.dataset}_{args.timesplit}_winlen_{args.window_len}_stepsize_{args.step_size}.csv'
    output_path = "./output/" + filename.split('.')[0]
    os.makedirs(output_path, exist_ok=True)
    
    df_preds = pd.DataFrame()
    for feature in args.features:
        df_pred = moving_prediction(args, feature)
        df_preds = pd.concat([df_preds, df_pred], axis=1)
        
    num_rows = len(df_preds)
    starts = list(range(0, num_rows * args.step_size, args.step_size))
    ends = [start + args.window_len for start in starts]
    
    # Ensure the number of generated rows matches df_preds
    if len(starts) > num_rows:
        starts = starts[:num_rows]
        ends = ends[:num_rows]

    df_preds['start'] = starts
    df_preds['end'] = ends
    
    df_preds.to_csv(os.path.join(output_path, filename), index=False)
    
    if args.save_plot:
        save_plots(args, output_path, df_preds)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    with open('./config/config.yaml', 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
        
    for key, value in config_data.items():
        parser.add_argument(f'--{key}', type=str, default=value, help=f'{key} argument from YAML')
    args = parser.parse_args()
    
    main(args)