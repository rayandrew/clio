import pandas as pd
import os

# Define input file path, output directory, and chunk size
input_file = '/mnt/nvme0n1/alibaba_block_traces_2020/device_124.parquet'
output_dir = '/mnt/nvme0n1/alibaba_block_traces_2020/device_124_chunks'
chunk_size = int(1e7)

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the Parquet file into a Pandas DataFrame
df = pd.read_parquet(input_file, engine='pyarrow')
print("DONE READING")

# Get the total number of rows
total_rows = len(df)
print(f"Total rows: {total_rows}")

# Calculate the number of chunks
num_chunks = int((total_rows // chunk_size) + (1 if total_rows % chunk_size != 0 else 0))
print(f"Number of chunks: {num_chunks}")

# Process and save chunks as CSV files
# make a file named header, and then append to it
header = df.columns
header_file = os.path.join(output_dir, "0.header")
# write
with open(header_file, 'w') as f:
    f.write(','.join(header) + '\n')


for i in range(num_chunks):
    start_row = i * chunk_size
    end_row = min((i + 1) * chunk_size, total_rows)
    chunk_df = df.iloc[start_row:end_row]

    # Define the output file path for the chunk
    output_file = os.path.join(output_dir, f"chunk_{i + 1}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the chunk to a CSV file
    chunk_df.to_csv(output_file, index=False, header=False)
    print(f"Saved chunk {i + 1}/{num_chunks} to {output_file}")
    if i == 4:
        break

print("Chunking completed.")
