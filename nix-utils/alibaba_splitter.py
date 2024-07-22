import pandas as pd
import os

# Input and output paths
input_file = '/mnt/nvme0n1/alibaba_block_traces_2020/device_124.parquet'
output_dir = './runs/raw/alibaba/split/124/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the Parquet file into a Pandas DataFrame
df = pd.read_parquet(input_file, engine='pyarrow')
print("done reading")

# Convert 'timestamp' from microseconds to milliseconds
first_timestamp = df['timestamp'].iloc[0]
df['timestamp'] = ((df['timestamp'] - first_timestamp)/ 1e3)

# Create 'minute' column
df['minute'] = (df['timestamp'] // (60 * 1e3)).astype(int)

# Map 'opcode' to 'io_type'
def map_opcode_to_io_type(opcode):
    return 1 if opcode == 'R' else 0

df['io_type'] = df['opcode'].apply(map_opcode_to_io_type)
print("Done mapping iotype", df.head())

# Reorder and rename columns
df = df[['timestamp', 'device_id', 'offset', 'length', 'io_type', 'minute']]
df.columns = ['ts_record', 'dummy', 'offset', 'size', 'io_type', 'minute']

print("SETTING INDICES")
# Set 'minute' as the index
df = df.set_index('minute')

# Compute the minimum and maximum minute values
max_minute = df.index.max()
min_minute = df.index.min()
print(f"Min minute: {min_minute}, Max minute: {max_minute}")

# Define a function to save each group to a CSV file
def save_chunk(group_df, idx):
    # Define the output path for the chunk
    output_path = f'{output_dir}/chunk_{idx}.csv'
    group_df.to_csv(output_path, index=False, header=False, sep=" ", columns=['ts_record', 'dummy', 'offset', 'size', 'io_type'])

# Group by 'minute' and save each group to a separate CSV file
for idx, minute in enumerate(range(min_minute, max_minute + 1)):
    print(f"Processing minute {minute}")
    minute_df = df.loc[[minute]].reset_index()
    save_chunk(minute_df, idx)
    print(f"Saved minute {minute}")
