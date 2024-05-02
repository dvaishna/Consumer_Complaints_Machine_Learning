import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StringType, TimestampType
from pyspark.sql.functions import year, month, dayofmonth, count, col, when, lit
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import pandas as pd

# PYSPARK_CLI = True
# if PYSPARK_CLI:
    # sc = SparkContext.getOrCreate()
    # spark = SparkSession(sc)


# Data Preprocessing
# ------------------

# Select necessary columns and drop corrupt records

# Read the JSON file 'complaints.json' into a DataFrame named 'raw_complaints'
raw_complaints = spark.read.json('5560_Complaints_DS/complaints.json')

# Select necessary columns and drop corrupt records
complaint_df = raw_complaints.select('company', 'product', 'timely', 'issue', 'state', 'date_sent_to_company').filter(raw_complaints['_corrupt_record'].isNull())

# ZipWithIndex and remove first row
complaint_df = complaint_df.rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0]).toDF()

# Drop rows with timely=None
df_timely_initial = complaint_df.filter(col("timely") != "")

# Cast date_sent_to_company to TimestampType
df_timely_initial = df_timely_initial.withColumn("date_sent_to_company", col("date_sent_to_company").cast(TimestampType()))

# Extract year, month, and day from date_sent_to_company
df_timely_initial = df_timely_initial.withColumn("year", year("date_sent_to_company")) \
                     .withColumn("month", month("date_sent_to_company")) \
                     .withColumn("day", dayofmonth("date_sent_to_company"))
                     
 
# Feature Engineering
# -------------------

# Calculate the frequency of each company
company_frequency = df_timely_initial.groupBy("company").agg(count("*").alias("frequency_company"))

# Join the frequency DataFrame with the original DataFrame on the company column
df_timely = df_timely_initial.join(company_frequency, on="company", how="left")

# Calculate the frequency of each issue
issue_frequency = df_timely_initial.groupBy("issue").agg(count("*").alias("frequency_issue"))

# Join the issue frequency DataFrame with the existing DataFrame on the issue column
df_timely = df_timely.join(issue_frequency, on="issue", how="left")

# Calculate the frequency of each state
state_frequency = df_timely_initial.groupBy("state").agg(count("*").alias("frequency_state"))

# Join the state frequency DataFrame with the existing DataFrame on the state column
df_timely = df_timely.join(state_frequency, on="state", how="left")

# Prepare Pipeline
# ----------------

# Define features_for_model
features_for_model = ["product", "frequency_company", "frequency_issue", "frequency_state"]

# Create a list of stages for the pipeline
stages = []

# Stage 1: String indexing for categorical features
indexer_product = StringIndexer(inputCol="product", outputCol="indexed_product")
stages.append(indexer_product)

# Stage 2: Assemble features
assembler = VectorAssembler(inputCols=["indexed_product", "frequency_company", "frequency_issue", "frequency_state", "year", "month", "day"], outputCol="assembledFeatures")
stages.append(assembler)

# Stage 3: String indexing for label
label_indexer = StringIndexer(inputCol="timely", outputCol="label")
stages.append(label_indexer)

# Stage 4: Linear SVM model
svm = LinearSVC(featuresCol="assembledFeatures", labelCol="label")
stages.append(svm)


# Balancing the data by Oversampling minority class (timely = No)
# ---------------------------------------------------------------

# Dropping additional columns
df_timely = df_timely.drop('company', 'issue', 'state', 'date_sent_to_company')

# Oversample the minority class
negative_df = df_timely.filter(col("timely") == "No")
balanced_ratio = df_timely.filter(col("timely") == "Yes").count() / negative_df.count()
oversampled_negative_df = negative_df.sample(withReplacement=True, fraction=balanced_ratio)
df_timely = df_timely.filter(col("timely") == "Yes").union(oversampled_negative_df)


# Model Training - Cross Validation
# ---------------------------------

# Persist the DataFrame in memory
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

# Define paramGrid
paramGrid = ParamGridBuilder() \
    .addGrid(svm.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(svm.maxIter, [5, 10, 15]) \
    .build()


# Create a CrossValidator
cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)

# Measure training time
import time
start_time = time.time()
model = cv.fit(train)
end_time = time.time()
training_time = end_time - start_time
minutes = int(training_time // 60)
seconds = int(training_time % 60)
logging.info("Training time: %02d:%02d" % (minutes, seconds))


# Model Evaluation - Cross Validation
# -----------------------------------

# Make predictions on the test set
predictions = model.transform(test)

# Calculate metrics
tp = float(predictions.filter("prediction == 1.0 AND label == 1").count())
fp = float(predictions.filter("prediction == 1.0 AND label == 0").count())
tn = float(predictions.filter("prediction == 0.0 AND label == 0").count())
fn = float(predictions.filter("prediction == 0.0 AND label == 1").count())
auc = evaluator.evaluate(predictions)

# Create DataFrame with evaluation metrics
metrics = spark.createDataFrame([
    ("TP", tp),
    ("FP", fp),
    ("TN", tn),
    ("FN", fn),
    ("Precision", round(tp / (tp + fp), 2)),
    ("Recall", round(tp / (tp + fn), 2)),
    ("AUC", round(auc, 2))
], ["metric", "value"])

# Print evaluation metrics
# print("*********CrossValidator Results *********")
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


# Print training time
logging.info("Training time: %02d:%02d" % (minutes, seconds))


# Model Evaluation - Train Validation Split
# -----------------------------------------

# Make predictions on the test data using the best model
predictions = tvModel.transform(test)

#Model Evaluate
accuracy = evaluator.evaluate(predictions)

# Extract TP, FP, TN, FN
tp = float(predictions.filter("prediction == 1.0 AND label == 1").count())
fp = float(predictions.filter("prediction == 1.0 AND label == 0").count())
tn = float(predictions.filter("prediction == 0.0 AND label == 0").count())
fn = float(predictions.filter("prediction == 0.0 AND label == 1").count())

# Calculate precision and recall
precision = round(tp / (tp + fp), 2)
recall = round(tp / (tp + fn), 2)

# Create DataFrame with evaluation metrics
metrics = spark.createDataFrame([
  ("TP", tp),
  ("FP", fp),
  ("TN", tn),
  ("FN", fn),
  ("Precision", precision),
  ("Recall", recall),
  ("AUC", round(accuracy, 2))  
], ["metric", "value"])
  
  
logging.info("***********TrainValidator Results ************")
metrics.show()