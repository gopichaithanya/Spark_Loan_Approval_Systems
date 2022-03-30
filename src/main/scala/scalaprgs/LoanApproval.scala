package scalaprgs

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


object LoanApproval extends App {

  Logger.getLogger("org.apache").setLevel(Level.ERROR)
  val spark = SparkSession.builder.master("local[3]").appName("LoanApp").getOrCreate()

  val file_location ="data//train_u6lujuX_CVtuZ9i.csv"
  val file_type ="csv"

  val infer_schema ="true"
  val first_row_is_header ="true"
  val delimiter =","

  val LoanDF = spark.read.format(file_type)
    .option("inferSchema",infer_schema)
    .option("header",first_row_is_header)
    .option("sep",delimiter)
    .load(file_location).na.fill(0)
    .na.fill("",Array("Gender"))
    .na.fill("",Array("Married"))
    .na.fill("0",Array("Dependents"))
    .na.fill("",Array("Self_Employed"))


  LoanDF.show(10)
  LoanDF.describe().show()
  LoanDF.printSchema()
  
  LoanDF.createOrReplaceTempView("LoanData")
  spark.sql("select count(Gender),count(Loan_Status),Gender,Loan_Status from LoanData group by Gender,Loan_Status").show()
  spark.sql("select count(Married),count(Loan_Status),Married,Loan_Status from LoanData group by Married,Loan_Status").show()
  spark.sql("select count(Dependents),count(Loan_Status),Dependents,Loan_Status from LoanData group by Dependents,Loan_Status sort by Dependents").show()
  spark.sql("select count(Education),count(Loan_Status),Education,Loan_Status from LoanData group by Education,Loan_Status").show()
  spark.sql("select count(Self_Employed),count(Loan_Status),Self_Employed,Loan_Status from LoanData group by Self_Employed,Loan_Status").show()
  spark.sql("select count(Property_Area),count(Loan_Status),Property_Area,Loan_Status from LoanData group by Property_Area,Loan_Status").show()
  spark.sql("select count(Credit_History),count(Loan_Status),Credit_History,Loan_Status from LoanData group by Credit_History,Loan_Status").show()
  spark.sql("select count(Loan_Amount_Term),count(Loan_Status),Loan_Amount_Term,Loan_Status from LoanData group by Loan_Amount_Term,Loan_Status").show()

  var StringfeatureCol = Array("Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status")
  val df = spark.createDataFrame( Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")) ).toDF("id", "category")
  val indexer = new StringIndexer() .setInputCol("category") .setOutputCol("categoryIndex")
  val indexed = indexer.fit(df).transform(df)
  indexed.show()
  val indexers = StringfeatureCol.map{colName=>new StringIndexer().setInputCol(colName).setOutputCol(colName+"_indexed")}
  val pipeline =new Pipeline().setStages(indexers)
  val LoanFinalDF = pipeline.fit(LoanDF).transform(LoanDF)

  LoanFinalDF.printSchema()

  val splits = LoanFinalDF.randomSplit(Array(0.7,0.3))
  val train = splits(0)
  val test = splits(1)
  val train_rows = train.count()
  val test_rows= test.count()
  println("Training Rows:"+ train_rows + " Testing Rows:" + test_rows)
  import spark.implicits._
  val assembler = new VectorAssembler()
    .setInputCols(Array("Loan_ID_indexed", "Gender_indexed", "Married_indexed", "Dependents_indexed", "Education_indexed", "Self_Employed_indexed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History","Property_Area_indexed")).setOutputCol("features")
  val training = assembler.transform(train).select($"features",$"Loan_Status_indexed".alias("label"))
  training.show(false)

  val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10)
    .setRegParam(0.3)
  val model = lr.fit(training)
  println("Model Trained!")

  val testing = assembler.transform(test).select($"features", $"Loan_Status_indexed".alias("trueLabel"))
  testing.show(false)

  val prediction = model.transform(testing)
  val predicted = prediction.select("features", "prediction", "trueLabel")

  predicted.show(1000)

  val evaluator = new BinaryClassificationEvaluator().setLabelCol("trueLabel").setLabelCol("trueLabel")
    .setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")

  val auc = evaluator.evaluate(prediction)

  println("AUC = " + (auc))

}