def flight(input, output):
    
	import pyspark
	from pyspark.sql import SparkSession
	from pyspark import SparkContext, SparkConf
	from pyspark.sql import SparkSession
	import pyspark.sql.functions as F
	from pyspark.sql import SQLContext
	from pyspark.sql.types import IntegerType, StringType
	from pyspark.ml import Pipeline
	from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
	from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator
	from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
	from pyspark.ml.clustering import KMeans
	
	# Dropping the null values from the dataset
	def drop_nan_values_spark(df):
		df_drop_nan_values_spark = df.na.drop()
		return df_drop_nan_values_spark
	
	# Merging the labels into Binary values to perform Binary Classification
	def merge_labels_spark(df):
		df_merge_labels_spark = df.select(df.CASE_SUBMITTED_YEAR, df.EMPLOYER_NAME, df.SOC_NAME, df.FULL_TIME_POSITION, df.PREVAILING_WAGE,
									df.WORKSITE_STATE, F.when(df.CASE_STATUS == "WITHDRAWN", "DENIED") \
														.when(df.CASE_STATUS == "CERTIFIEDWITHDRAWN", "CERTIFIED") \
														.otherwise(df.CASE_STATUS).alias("CASE_STATUS"))
		return df_merge_labels_spark
	
	# Dividing the wages into the ranges 
	def prevailing_wage_spark(df):
		df_prevailing_wage_spark = df.select(df.CASE_SUBMITTED_YEAR, df.EMPLOYER_NAME, df.SOC_NAME, df.FULL_TIME_POSITION,
										F.when(df.PREVAILING_WAGE <= 20000, "0-20000") \
										.when((df.PREVAILING_WAGE > 20000) & (df.PREVAILING_WAGE <= 50000), "20000-50000") \
										.when((df.PREVAILING_WAGE > 50000) & (df.PREVAILING_WAGE <= 120000), "50000-120000") \
										.when((df.PREVAILING_WAGE > 120000) & (df.PREVAILING_WAGE <= 250000), "120000-250000") \
										.otherwise(">250000").alias("WAGE_RANGE"), df.WORKSITE_STATE, df.CASE_STATUS)
		return df_prevailing_wage_spark
	
	# Changing the individual field to its Industry field
	def classify_employer_spark(df):
		df_classify_employer_spark = df.select(df.CASE_SUBMITTED_YEAR, df.EMPLOYER_NAME,
										F.when((df.SOC_NAME == "COMPUTER OCCUPATION") | (df.SOC_NAME == "GRAPHIC DESIGNERS") | 
												(df.SOC_NAME == "ANALYSTS"), "IT INDUSTRY") \
										.when((df.SOC_NAME == "ACCOUNTANTS") | (df.SOC_NAME == "BUSINESS OPERATIONS SPECIALIST") | 
												(df.SOC_NAME == "CHIEF EXECUTIVES") | (df.SOC_NAME == "CURATORS") | 
												(df.SOC_NAME == "EVENT PLANNERS") | (df.SOC_NAME == "FIRST LINE SUPERVISORS") | 
												(df.SOC_NAME == "HUMAN RESOURCES") | (df.SOC_NAME == "IT MANAGERS") | 
												(df.SOC_NAME == "MANAGEMENT") | (df.SOC_NAME == "MANAGERS") | 
												(df.SOC_NAME == "PUBLIC RELATIONS"), "MANAGEMENT") \
										.when((df.SOC_NAME == "ACTUARIES") | (df.SOC_NAME == "FINANCE"), "FINANCE") \
										.when((df.SOC_NAME == "AGRICULTURE") | (df.SOC_NAME == "ANIMAL HUSBANDARY") | 
												(df.SOC_NAME == "FOOD PREPARATION WORKERS"), "FOOD AND AGRICULTURE") \
										.when((df.SOC_NAME == "COACHES AND SCOUTS") | (df.SOC_NAME == "COUNSELORS") | 
												(df.SOC_NAME == "EDUCATION")| (df.SOC_NAME == "FITNESS TRAINERS") | 
												(df.SOC_NAME == "INTERPRETERS AND TRANSLATORS") | (df.SOC_NAME == "LIBRARIANS") | 
												(df.SOC_NAME == "LOGISTICIANS") | (df.SOC_NAME == "SURVEYORS") | 
												(df.SOC_NAME == "WRITERS EDITORS AND AUTHORS"), "EDUCATION") \
										.when((df.SOC_NAME == "SALES AND RELATED WORKERS") | (df.SOC_NAME == "MARKETING"), "MARKETING") \
										.when((df.SOC_NAME == "DOCTORS") | (df.SOC_NAME == "SCIENTIST") | 
												(df.SOC_NAME == "INTERNIST"), "ADVANCED SCIENCES") \
										.when((df.SOC_NAME == "COMMUNICATIONS") | (df.SOC_NAME == "ENGINEERS") | 
												(df.SOC_NAME == "LAB TECHNICIANS") | (df.SOC_NAME == "CONSTRUCTION") | 
												(df.SOC_NAME == "ARCHITECTURE") | (df.SOC_NAME == "MECHANICS"), "ENGINEERING AND ARCHITECTURE") \
										.otherwise("ARTISTS AND ENTERTAINMENT").alias("INDUSTRY"), df.FULL_TIME_POSITION, df.WAGE_RANGE, df.WORKSITE_STATE, df.CASE_STATUS)
		return df_classify_employer_spark
		
	# Implementation of the Spark Code
	spark = SparkSession.builder.getOrCreate()
	sc = spark.read
	sc.option('header', True)
	sc.option('inferSchema', True)
	sqlContext = SQLContext(spark)

    # Creating the dataframe from the CSV file
	df_H1b_file = sc.csv(input)
    
	# Pre-processing the data
	df_dnv = drop_nan_values_spark(df_H1b_file)
	df_ml = merge_labels_spark(df_dnv)
	df_pw = prevailing_wage_spark(df_ml)
	df_ce = classify_employer_spark(df_pw)
	
	# DataFrame after pre-processing the data from the original dataframe
	print("Data after applying pre-processing methods:")
	df_ce.show(10)
	
	# Converting the values using StringIndexer and encoding the values with OneHotEncoder
	categoricalColumns = ["EMPLOYER_NAME", "INDUSTRY", "FULL_TIME_POSITION", "WAGE_RANGE", "WORKSITE_STATE"]
	stages = [] # stages in the Pipeline
	for categoricalCol in categoricalColumns:
		# Category Indexing with StringIndexer
		stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
		# Encoding the values
		encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
		# Adding the stages
		stages += [stringIndexer, encoder]
	
	# Setting the label value from CASE_STATUS which is to be predicted 
	label_stringIdx = StringIndexer(inputCol="CASE_STATUS", outputCol="label")
	stages += [label_stringIdx]
	
	# Using the VectorAssembler to get the labels vector for the prediction 
	assemblerInputs = [c + "classVec" for c in categoricalColumns] + ["CASE_SUBMITTED_YEAR"]
	assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
	stages += [assembler]
	
	# Implementing the pipeline for the flow
	partialPipeline = Pipeline().setStages(stages)
	pipelineModel = partialPipeline.fit(df_ce)
	preppedDataDF = pipelineModel.transform(df_ce)
	
	selectedcols = ["label", "features"] + df_ce.columns
	final_dataset = preppedDataDF.select(selectedcols) # DataFrame to be used for the Machine Learning Models
	
	# Dividing the dataset into training and testing samples
	(trainData, testData) = final_dataset.randomSplit([0.7, 0.3], seed=100)
	print("Number of samples to train the model: " + str(trainData.count()))
	print("Number of samples to test the model: " + str(testData.count()))
	
	# Calling the Logistic Regression Model from the MLlib in Spark
	lrModel = LogisticRegression(featuresCol= 'features', labelCol= 'label', maxIter= 15)
	
	# Fitting the training data in the model to train the data
	LR_Model = lrModel.fit(trainData)
	
	# Predicting the outputs for the test data
	predictions_LR = LR_Model.transform(testData)
	
	print("Predictions analysis for Logistic Regression Model:")
	predictions_LR.select("EMPLOYER_NAME", "INDUSTRY", "FULL_TIME_POSITION", "WAGE_RANGE", "WORKSITE_STATE", "label", "rawPrediction", "prediction", "probability").show(10)
	
	# Evaluating the accuracy for the Logistic Regression
	evaluator = BinaryClassificationEvaluator()
	LR_accuracy = str(evaluator.evaluate(predictions_LR, {evaluator.metricName: "areaUnderROC"}))
	print("Accuracy for Logistic Regression Model: " + LR_accuracy)
	
	# Defining the accuracy list to store the accuracies of the model
	accuracy = []
	
	# Appending the accuracy of Logistic Regression Model
	accuracy.append(LR_accuracy)
	
	# Implementation of Random Forest Model
	rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
	
	rfModel = rf.fit(trainData)
	
	predictions = rfModel.transform(testData)
	
	print("Predictions analysis for Random Forest Model:")
	predictions_LR.select("EMPLOYER_NAME", "INDUSTRY", "FULL_TIME_POSITION", "WAGE_RANGE", "WORKSITE_STATE", "label", "rawPrediction", "prediction", "probability").show(10)
	
	evaluator = BinaryClassificationEvaluator()
	RF_accuracy = str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
	print("Accuracy for Random Forest Model: " + RF_accuracy)
	
	# Appending the accuracy of Random Forest Model
	accuracy.append(RF_accuracy)
	
	# Converting the list to a Dataframe
	df_accuracy = sqlContext.createDataFrame(accuracy, StringType())
	df_accuracy=df_accuracy.selectExpr("Value as Accuracy")
	
	models=["Logistic Regression Model","Random Forest Model"]
	df_models = sqlContext.createDataFrame(models, StringType())
	df_models=df_models.selectExpr("Value as Models")
	
	df_accuracy=df_accuracy.withColumn("id",F.monotonically_increasing_id())
	df_models=df_models.withColumn("id",F.monotonically_increasing_id())
	df_final=df_models.join(df_accuracy,"id","outer").drop("id")
	df_final.show()
		
	# Writing the file back to the storage
	df_final.repartition(1).write.option("header","true").format('csv').save(output)
	
	# Implementation of kMeans Model
	for k in range(2,9):
		kmeans = KMeans(featuresCol= "features", k=k)
		model = kmeans.fit(trainData)
		wsse = model.computeCost(trainData)
		print("k = {}, the error is {}".format(k,str(wsse))) # Showing the Squared Sum Errors for different values of k
	
	spark.stop()
    
def files_from_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='input')
    parser.add_argument('-o', '--output',default='output')
    args = parser.parse_args()
    return (args.input, args.output)

if __name__ == "__main__":
    inputfile, outputfile = files_from_args()
    flight(inputfile, outputfile)