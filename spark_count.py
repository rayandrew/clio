from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType

# Create a Spark session
spark = SparkSession.builder \
    .appName("Large CSV Processing") \
    .getOrCreate()


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

# df.write.parquet("./alibaba/parquet")

# Group by 'device_id' and count the number of rows for each device
device_io_counts = df.groupBy("device_id").count()

device_io_counts.write.csv('./alibaba/device_io_counts.csv')
