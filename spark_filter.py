from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType

DEVICE_ID = 277

# Create a Spark session
spark = SparkSession.builder \
    .appName("Large CSV Processing") \
    .getOrCreate()

spark.sparkContext.setLogLevel("OFF")

# Define the CSV file path
csv_file_path = '/mnt/nvme0n1/alibaba_block_traces_2020/io_traces.csv'

# Define the schema
schema = StructType([
    StructField("device_id", IntegerType(), True),
    StructField("opcode", StringType(), True),
    StructField("offset", LongType(), True),
    StructField("length", IntegerType(), True),
    StructField("timestamp", LongType(), True)
])

# Read the CSV file into a DataFrame with the schema
df = spark.read.csv(csv_file_path, schema=schema, header=False)

filtered_df = df.filter(df["device_id"] == DEVICE_ID)

# 

# Define the output path for the Parquet file
output_parquet_path = f"./alibaba/parquet/{DEVICE_ID}"

# Write the filtered DataFrame to Parquet format
filtered_df.write.mode('overwrite').parquet(output_parquet_path)

## CSV, tar.gzip

