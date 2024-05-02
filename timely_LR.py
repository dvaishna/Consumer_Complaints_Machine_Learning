import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import StringType
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import pandas as pd

# PYSPARK_CLI = True
# if PYSPARK_CLI:
    # sc = SparkContext.getOrCreate()
    # spark = SparkSession(sc)

# Data Preprocessing
# ------------------

# Read the JSON file 'complaints.json' into a DataFrame named 'raw_complaints'
raw_complaints = spark.read.json('5560_Complaints_DS/complaints.json')

# Select necessary columns and drop corrupt records
complaint_df = raw_complaints.select('company', 'product', 'timely' , 'issue', 'state', 'date_sent_to_company').filter(raw_complaints['_corrupt_record'].isNull())

complaint_df = complaint_df.rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0]).toDF()

# drop 1 row having timely = None and Create a dataframe for prediction of timely_response
df_timely_initial = complaint_df.filter(col("timely") != "")

# Cast date_sent_to_company to a suitable type 'timestamp'
df_timely_initial = df_timely_initial.withColumn("date_sent_to_company", col("date_sent_to_company").cast(TimestampType()))

# Extracting year, month, and day from 'date_sent_to_company' column
df_timely_initial = df_timely_initial.withColumn("year", year("date_sent_to_company")) \
                     .withColumn("month", month("date_sent_to_company")) \
                     .withColumn("day", dayofmonth("date_sent_to_company"))
          
          
# Feature Engineering
# -------------------            

# Calculate the frequency of each company
company_frequency = df_timely_initial.groupBy("company").agg(count("*").alias("frequency_company"))

# Join the frequency DataFrame with the original DataFrame on the company column
df_timely = df_timely_initial.join(company_frequency, on="company", how="left")

# Calculate the frequency of each issue (corrected to avoid duplicate calculation)
issue_frequency = df_timely_initial.groupBy("issue").agg(count("*").alias("frequency_issue"))

# Join the issue frequency DataFrame with the existing DataFrame on the issue column
df_timely = df_timely.join(issue_frequency, on="issue", how="left")

# Calculate the frequency of each issue (corrected to avoid duplicate calculation)
state_frequency = df_timely_initial.groupBy("state").agg(count("*").alias("frequency_state"))

# Join the issue frequency DataFrame with the existing DataFrame on the issue column
df_timely = df_timely.join(state_frequency, on="state", how="left")


# Prepare Pipeline
# ----------------

# Define features_for_model directly with data types
features_for_model = ["product", "frequency_company", "frequency_issue", "frequency_state"]

# Create a list of stages for the pipeline
stages = []

# String indexing for categorical features
indexer_product = StringIndexer(inputCol="product", outputCol="indexed_product")

# Stage 1: Append all indexers to the stages list
stages.append(indexer_product)

# Create the VectorAssembler instance
assembler = VectorAssembler(inputCols=["indexed_product", "frequency_company", "frequency_issue", "frequency_state", "year", "month", "day"], outputCol="assembledFeatures")

# Stage 2: Assemble features
stages.append(assembler)

# Stage 3: String indexing for label
label_indexer = StringIndexer(inputCol="timely", outputCol="label")
stages.append(label_indexer)

# Stage 4: Logistic Regression model
lr = LogisticRegression(featuresCol="assembledFeatures", labelCol="label")
stages.append(lr)

# Balancing the data by Oversampling minority class (timely = No)
# ---------------------------------------------------------------

# dropping additional columns
df_timely = df_timely.drop('company' , 'issue', 'state', 'date_sent_to_company')
df_timely.show(10)

# Oversample the minority class (assuming "No" is the minority)
negative_df = df_timely.filter(col("timely") == "No")

# Calculate the fraction to achieve a more balanced ratio
# For example, if you want a 1:1 ratio, set fraction = number of "Yes" instances / number of "No" instances
balanced_ratio = df_timely.filter(col("timely") == "Yes").count() / negative_df.count()
oversampled_negative_df = negative_df.sample(withReplacement=True, fraction=balanced_ratio)

# Combine oversampled negatives with original data (assuming positive is the majority)
df_timely = df_timely.filter(col("timely") == "Yes").union(oversampled_negative_df)


# def calculate_weights(data, label_column, weight_column, weight_value):
    # total_count = data.count()
    # positive_count = data.filter(col(label_column) == "Yes").count()
    # negative_count = data.filter(col(label_column) == "No").count()
    # weight_positive = lit(total_count) / (2 * positive_count * weight_value)
    # weight_negative = lit(total_count) / (2 * negative_count * (1 - weight_value))
    # return data.withColumn(weight_column, when(col(label_column) == "Yes", weight_positive).otherwise(weight_negative))

# # Calculate weights for the minority class
# df_timely_balanced = calculate_weights(df_timely, "timely", "weight", 0.3)

# Model Training - Cross Validation
# ---------------------------------

from pyspark.storagelevel import StorageLevel
df_timely.persist(StorageLevel.MEMORY_ONLY)

# Split data into training and testing sets
train, test = df_timely.randomSplit([0.7, 0.3], seed=42)

# Print the number of rows in train and test DataFrames
logging.info("Number of rows in train DataFrame: {}".format(train.count()))
logging.info("Number of rows in test DataFrame: {}".format(test.count()))

# Combine stages into a pipeline
pipeline = Pipeline(stages=stages)

# Define evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

#Define paramGrid
paramGrid = ParamGridBuilder() \
 .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
 .build()

# Create a CrossValidator
cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)

import time
# Start time
start_time = time.time()

# Fit the model with cross-validation on the training set
model = cv.fit(train)

# End time
end_time = time.time()

# Calculate training time
training_time = end_time - start_time


# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

logging.info("Training time: %02d:%02d" % (minutes, seconds))


# Model Evaluation - Cross Validation
# -----------------------------------

# Make predictions on the test set (use the actual test set)
predictions = model.transform(test)

tp = float(predictions.filter("prediction == 1.0 AND label == 1").count())
fp = float(predictions.filter("prediction == 1.0 AND label == 0").count())
tn = float(predictions.filter("prediction == 0.0 AND label == 0").count())
fn = float(predictions.filter("prediction == 0.0 AND label == 1").count())

auc = evaluator.evaluate(predictions)

metrics = spark.createDataFrame([
    ("TP", tp),
    ("FP", fp),
    ("TN", tn),
    ("FN", fn),
    ("Precision", tp / (tp + fp)),
    ("Recall", tp / (tp + fn)),
    ("AUC", auc)
], ["metric", "value"])

metrics = metrics.withColumn("value", round(col("value"), 2))

logging.info("*********CrossValidator Results *********")
metrics.show()


#Train Validation Split
#----------------------

from pyspark.ml.tuning import TrainValidationSplit
from pyspark.sql.functions import col

# Define TrainValidationSplit
trainval = TrainValidationSplit(estimator=pipeline,
                                 estimatorParamMaps=paramGrid,
                                 evaluator=evaluator,
                                 trainRatio=0.8) 
                                 
                                    
#Training the model and Calculating its time
import time

# Start time
start_time = time.time() 

# Fit the cross validator to the training data
tvModel = trainval.fit(train)

# End time
end_time = time.time()

print("Model trained!")


# Calculate training time
training_time = end_time - start_time

# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

logging.info("Training time: %02d:%02d" % (minutes, seconds))


# Model Evaluation - Train Validation Split
# ----------------------------------------

# Make predictions on the test data using the best model
predictions = tvModel.transform(test)

#Model Evaluate
accuracy = evaluator.evaluate(predictions)

# Extract TP, FP, TN, FN
tp = float(predicted.filter("prediction == 1.0 AND label == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND label == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND label == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND label == 1").count())

metrics = spark.createDataFrame([
    ("TP", tp),
    ("FP", fp),
    ("TN", tn),
    ("FN", fn),
    ("Precision", tp / (tp + fp)),
    ("Recall", tp / (tp + fn)),
    ("AUC", auc)
], ["metric", "value"])

metrics = metrics.withColumn("value", round(col("value"), 2))
  
logging.info("***********TrainValidator Results ************")
metrics.show()
