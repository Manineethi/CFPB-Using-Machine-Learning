# -*- coding: utf-8 -*-
"""This Spark code performs classification on a complaints dataset 
using a Random Forest model trained with CrossValidator for hyperparameter tuning.
 
It calculates feature importances, evaluates model performance 
using various metrics (accuracy, precision, recall), 
and compares the results between CrossValidator and TrainValidationSplit approaches.

Data Loading and Preprocessing:
Feature Engineering:
Feature Selection and Preprocessing:
Model Building:"""



from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import pandas as pd

from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql.functions import lit
from pyspark.ml import Transformer
from pyspark.sql.functions import col, isnull, trim
from pyspark.ml.feature import IndexToString
from pyspark.ml.feature import VectorAssembler, StringIndexer

# PYSPARK_CLI = True
# if PYSPARK_CLI:
    # sc = SparkContext.getOrCreate()
    # spark = SparkSession(sc)


#Data Loading and Preprocessing:
"""Read the JSON file 'complaints.json' into a DataFrame named 'raw_complaints'.
Select necessary columns ('company', 'product', 'company_response', 'issue').
Filter out corrupt records based on the '_corrupt_record' column.
Remove rows with missing or empty values in 'company', 'product', or 'company_response'."""


# Read the JSON file 'complaints.json' into a DataFrame named 'raw_complaints'
raw_complaints = spark.read.json('/user/dvaishn2/5560_Complaints_DS/complaints.json')

# Select necessary columns and drop corrupt records
complaint_df = raw_complaints.select('company', 'product', 'company_response' , 'issue').filter(raw_complaints['_corrupt_record'].isNull())

complaint_df = complaint_df.filter(~(isnull(col("company")) | (trim(col("company")) == "")))
complaint_df = complaint_df.filter(~(isnull(col("product")) | (trim(col("product")) == "")))
complaint_df = complaint_df.filter(~(isnull(col("company_response")) | (trim(col("company_response")) == "")))

# Show the first 10 rows of the DataFrame 'complaint_df'
complaint_df.show(10)

# Load dataset (assuming `complaint_df` is already defined)
df_company_response = complaint_df



#Feature Engineering:
"""Calculate frequency of each company (company_frequency).
Calculate frequency of each issue (issue_frequency).
Join the frequency DataFrames with the original DataFrame on 'company' and 'issue' columns, respectively."""



# Calculate the frequency of each company
company_frequency = df_company_response.groupBy("company").agg(count("*").alias("frequency_company"))

# Join the frequency DataFrame with the original DataFrame on the company column
df_response_with_frequency = df_company_response.join(company_frequency, on="company", how="left")

# Calculate the frequency of each issue (corrected to avoid duplicate calculation)
issue_frequency = df_company_response.groupBy("issue").agg(count("*").alias("frequency_issue"))

# Join the issue frequency DataFrame with the existing DataFrame on the issue column
df_response_with_frequency = df_response_with_frequency.join(issue_frequency, on="issue", how="left")

# Show the result
df_response_with_frequency.show(10)



#Feature Selection and Preprocessing:
"""Define features (product, frequency_company, frequency_issue) and target (company_response).
Perform string indexing for the target variable and product using StringIndexer.
Create a VectorAssembler to combine indexed features into a single feature vector."""


# Use the frequency column as a feature for modeling
features = ["product", "frequency_company", "frequency_issue"] 
target = "company_response"


from pyspark.storagelevel import StorageLevel
 
df_response_with_frequency.persist(StorageLevel.MEMORY_ONLY)
  
# String indexing for target variable
target_indexer = StringIndexer(inputCol="company_response", outputCol="indexed_company_response")

indexer_product = StringIndexer(inputCol="product", outputCol="indexed_product")

df_response_with_frequency = df_response_with_frequency.drop('company' , 'issue')

df_response_with_frequency.show(10)

# Create VectorAssembler to combine the indexed product and hashed company features
assembler = VectorAssembler(inputCols=["indexed_product", "frequency_company", "frequency_issue"], outputCol="features")

# Create Random Forest model
rf = RandomForestClassifier(labelCol="indexed_company_response", featuresCol="features")

# Create a pipeline with the VectorAssembler and Random Forest model
pipeline = Pipeline(stages=[indexer_product, target_indexer, assembler, rf])

# Split the data into training and testing sets
train_data, test_data = df_response_with_frequency.randomSplit([0.7, 0.3], seed=42)

train_rows = train_data.count()
test_rows = test_data.count()

# Print the counts
print("Training Rows:", train_rows, " Testing Rows:", test_rows)



#Model Building:
"""Create a Random Forest model (RandomForestClassifier).
Define a pipeline comprising the feature preprocessing stages and the model.
Set up a ParamGridBuilder for hyperparameter tuning"""



evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexed_company_response", metricName="accuracy")

# Define parameter grid for hyperparameter tuning                     
                         
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20]) \
    .addGrid(rf.maxDepth, [2, 4, 6]) \
    .addGrid(rf.minInstancesPerNode, [1, 5, 10]) \
    .build()

# Define CrossValidator
crossval = CrossValidator(estimator=pipeline,
                                    estimatorParamMaps=paramGrid,
                                    evaluator=evaluator,
                                    numFolds=3)

from pyspark.sql.functions import col

                                    
#Training the model and Calculating its time
import time

# Start time
start_time = time.time() 

# Fit the cross validator to the training data
cvModel = crossval.fit(train_data)

# End time
end_time = time.time()

print("Model trained!")

# Calculate training time
training_time = end_time - start_time

# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

# Format the time
training_time_formatted = "{:02d}:{:02d}".format(minutes, seconds)

# Print training time
print("Training time:", training_time_formatted)

# Feature Importance:

# Get the fitted model from CrossValidator
bestModel = cvModel.bestModel

# Access the feature importances from the Random Forest model within the pipeline
feature_importances = bestModel.stages[-1].featureImportances  # Assuming Random Forest is the last stage

# Get feature names from the VectorAssembler
feature_names = assembler.getInputCols()

# Create a DataFrame of feature importances
featureImp = pd.DataFrame(list(zip(feature_names, feature_importances)), columns=["feature", "importance"])

# Sort the DataFrame by importance (descending order)
featureImp = featureImp.sort_values(by="importance", ascending=False)

# Print the DataFrame with feature importance
print("\nFeature Importance:")
print(featureImp.to_string(index=False))
 
#Test the Data : 


# Make predictions on the test data using the best model
predictions = cvModel.transform(test_data)



# # Evaluate model performance
# accuracy_rf = evaluator.evaluate(predictions)
# print("Accuracy (Random Forest):", accuracy_rf)

# # Define the evaluator for precision
# precision_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexed_company_response", metricName="weightedPrecision")

# # Calculate precision
# precision_rf = precision_evaluator.evaluate(predictions)
# print("Precision (Random Forest):", precision_rf)

# # Define the evaluator for recall
# recall_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexed_company_response", metricName="weightedRecall")

# # Calculate recall
# recall_rf = recall_evaluator.evaluate(predictions)
# print("Recall (Random Forest):", recall_rf)

# # Create DataFrame with evaluation metrics
# Values = spark.createDataFrame([
  # ("Accuracy", accuracy_rf),
  # ("Precision", precision_rf),
  # ("Recall", recall_rf)  
# ], ["metric", "value"])
  
# print ("***CrossValidator Results ****")
# Values.show()



# Evaluate the model
accuracy = evaluator.evaluate(predictions)

# Extract TP, FP, TN, FN
tp = float(predictions.filter("prediction == 1.0 AND indexed_company_response == 1").count())
fp = float(predictions.filter("prediction == 1.0 AND indexed_company_response == 0").count())
tn = float(predictions.filter("prediction == 0.0 AND indexed_company_response == 0").count())
fn = float(predictions.filter("prediction == 0.0 AND indexed_company_response == 1").count())

# Calculate precision and recall
precision = tp / (tp + fp)
recall = tp / (tp + fn)

# Create DataFrame with evaluation metrics
metrics = spark.createDataFrame([
  ("TP", tp),
  ("FP", fp),
  ("TN", tn),
  ("FN", fn),
  ("Accuracy", accuracy),
  ("Precision", precision),
  ("Recall", recall)  
], ["metric", "value"])
  
print ("***CrossValidator Results ****")
metrics.show()


#Train Validation

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
tvModel = trainval.fit(train_data)

# End time
end_time = time.time()

print("Model trained!")


# Calculate training time
training_time = end_time - start_time

# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

# Format the time
training_time_formatted = "{:02d}:{:02d}".format(minutes, seconds)

# Print training time
print("Training time:", training_time_formatted)

# Make predictions on the test data using the best model
predictions = tvModel.transform(test_data)

#Model Evaluate
accuracy = evaluator.evaluate(predictions)

# Extract TP, FP, TN, FN
tp = float(predictions.filter("prediction == 1.0 AND indexed_company_response == 1").count())
fp = float(predictions.filter("prediction == 1.0 AND indexed_company_response == 0").count())
tn = float(predictions.filter("prediction == 0.0 AND indexed_company_response == 0").count())
fn = float(predictions.filter("prediction == 0.0 AND indexed_company_response == 1").count())

# Calculate precision and recall
precision = tp / (tp + fp)
recall = tp / (tp + fn)

# Create DataFrame with evaluation metrics
metrics = spark.createDataFrame([
  ("TP", tp),
  ("FP", fp),
  ("TN", tn),
  ("FN", fn),
  ("Accuracy", accuracy),
  ("Precision", precision),
  ("Recall", recall)  
], ["metric", "value"])
  
print ("***TrainValidator Results ****")
metrics.show()


# Evaluate model performance
accuracy_rf = evaluator.evaluate(predictions)

# Define the evaluator for precision
precision_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexed_company_response", metricName="weightedPrecision")

# Calculate precision
precision_rf = precision_evaluator.evaluate(predictions)

# Define the evaluator for recall
recall_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexed_company_response", metricName="weightedRecall")

# Calculate recall
recall_rf = recall_evaluator.evaluate(predictions)

# Create DataFrame with evaluation metrics
Values = spark.createDataFrame([
  ("Accuracy", accuracy_rf),
  ("Precision", precision_rf),
  ("Recall", recall_rf)  
], ["metric", "value"])
  
print ("***TrainValidator Results ****")
Values.show()

                                 

