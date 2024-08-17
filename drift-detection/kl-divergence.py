import yaml
import argparse
import pandas as pd
import numpy as np
import os, datetime
import matplotlib.pyplot as plt
from scipy.spatial import distance

from numpy.fft import *

def moving_average(signal, window_size):
    # Create a window of the given size
    window = np.ones(window_size) / window_size
    
    # Apply convolution between the signal and the window
    filtered_signal = np.convolve(signal, window, mode='same')
    
    return filtered_signal

def filter_signal(signal, threshold=1e8):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)

def preprocess(args, data):
    
    if args.preprocess == 'none':
        return data
    elif args.preprocess == 'fft':
        filtered_data = filter_signal(data, threshold=40000)
    elif args.preprocess == 'mean':
        filtered_data = moving_average(data, window_size=10)
        
    return filtered_data

def save_plots(args, output_path, df_preds):
    
    df_data = pd.read_csv(args.data_path)
    
    if df_preds.columns.duplicated().any():
        raise ValueError("Duplicate column names found in df_preds.")
    
    for feature in args.features:
        
        for drift_detector in ['kl']:
        
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
                if args.preprocess == 'none':
                    save_path = f"{output_path}/{feature}_label_{label}_{drift_detector}.png"
                else:
                    save_path = f"{output_path}/{feature}_label_{label}_{drift_detector}_{args.preprocess}.png"
                plt.savefig(save_path)
                plt.close()  # Close the figure after saving to free up memory

                print(f"Plot saved to {save_path}")
    
    return

def process_preds(feature, preds):
    
    preds = pd.DataFrame(preds, columns=[f'{feature}_kl_distance'])
    preds[f'{feature}_kl_label'] = 0
    sorted_df = preds.sort_values(by=f'{feature}_kl_distance', ascending=False)
    lowest_ten_indices = sorted_df.head(10).index
    highest_ten_indices = sorted_df.tail(10).index
    preds.loc[lowest_ten_indices, f'{feature}_kl_label'] = 1
    preds.loc[highest_ten_indices, f'{feature}_kl_label'] = 2
    
    # Counting the number of labels for KS
    label_counts = preds[f'{feature}_kl_label'].value_counts()
    print(f"kl-divergence Labels for {feature}:")
    print(label_counts)
    
    return preds

def normalize_distribution(p):
    total = np.sum(p)
    return p / total

def kl_divergence(p, q):
    
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    return np.sum(p * np.log(p / q))
    

def moving_prediction(args, feature):
    
    df_data = pd.read_csv(args.data_path)
    data = preprocess(args, df_data[feature].to_numpy())
    
    window_len = args.window_len
    step_size = args.step_size
    ref_start, ref_end = 0, window_len
    start, end = window_len, window_len + window_len
    preds = []
    
    while end < len(df_data):
        data_ref = data[ref_start:ref_end]
        x = data[start:end]
        
        kl = kl_divergence(data_ref, x)
        
        preds.append(kl)
        
        ref_start += step_size
        ref_end = ref_start + window_len
        start += step_size
        end = start + window_len
        
    df_preds = process_preds(feature, preds)
    
    return df_preds

def main(args):
    
    filename = f'{args.dataset}_{args.timesplit}_winlen_{args.window_len}_stepsize_{args.step_size}'
    output_path = "./output/" + filename
    if args.preprocess == 'none':
        filename = filename + '_kl-divergence.csv'
    else:
        filename = filename + '_kl-divergence' + '_' + args.preprocess + '.csv'
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