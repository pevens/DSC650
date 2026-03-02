from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create a Spark session
spark = SparkSession.builder.appName("MLlib IQ ML Prediction").enableHiveSupport().getOrCreate()

# Step 2: Load the data from the Hive table 'iq' into a Spark DataFrame
iq_df = spark.sql("SELECT literacy_rate, nobel_prices, hdi_2021, mean_years_schooling_2021,gni_2021, population_2023, average_iq FROM iq")

print(f"DEBUG: Initial Row Count from Hive: {iq_df.count()}")
iq_df.show(5) # This will show if the columns are correctly mapped

# Step 3: Handle null values by either dropping or filling them
iq_df = iq_df.na.drop()  # Drop rows with null values

# Step 4: Prepare the data for MLlib by assembling features into a vector
assembler = VectorAssembler(
    inputCols=["literacy_rate", "nobel_prices", "hdi_2021", "mean_years_schooling_2021","gni_2021", "population_2023"],
    outputCol="features",
    handleInvalid="skip"  # Skip rows with null values
)
assembled_df = assembler.transform(iq_df).select("features", "average_iq")

# Step 5: Split the data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

# Step 6: Initialize and train a Linear Regression model
lr = LinearRegression(labelCol="average_iq")
lr_model = lr.fit(train_data)

# Step 7: Evaluate the model on the test data
test_results = lr_model.evaluate(test_data)

# Step 8: Print the model performance metrics
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# ---- Write metrics to HBase with happybase (using the provided pattern) ----
# Example data (row_key, column_family:column, value) populated with the metrics
data = [
    ('metrics1', 'cf:rmse', str(test_results.rootMeanSquaredError)),
    ('metrics1', 'cf:r2',   str(test_results.r2)),
]

# Function to write data to HBase inside each partition
def write_to_hbase_partition(partition):
    try:
        connection = happybase.Connection('master')
        connection.open()
        table = connection.table('my_table')  # Update table name
        for row in partition:
            row_key, column, value = row
            table.put(row_key, {column: value})
        connection.close()
    except Exception as e:
        print(f"ERROR writing to HBase: {e}")

# Parallelize data and apply the function with foreachPartition
rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 9: Stop the Spark session
spark.stop()
