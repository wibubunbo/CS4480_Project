import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.tartarus.snowball.ext.PorterStemmer
import edu.stanford.nlp.simple.Document

object DataCleaning {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Amazon Review Data Cleaning")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Read the review dataset
    val df = spark.read.json("Cell_Phones_and_Accessories.json")

    // Select only the columns we need for sentiment analysis
    val selectedDF = df.select("overall", "reviewText")

    // Clean the text data with three versions
    val cleanedDF = cleanText(selectedDF)

    // Filter out reviews with only one word for each version
    val filteredDF1 = cleanedDF.filter(size(split($"cleaned_reviewText", " ")) > 1)
    val filteredDF2 = cleanedDF.filter(size(split($"cleaned_reviewText_stemmed", " ")) > 1)
    val filteredDF3 = cleanedDF.filter(size(split($"cleaned_reviewText_lemmatized", " ")) > 1)

    // Create balanced datasets
    val balancedDF1 = createBalancedDataset(filteredDF1, "cleaned_reviewText")
    val balancedDF2 = createBalancedDataset(filteredDF2, "cleaned_reviewText_stemmed")
    val balancedDF3 = createBalancedDataset(filteredDF3, "cleaned_reviewText_lemmatized")

    // Split into train and test sets
    val Array(trainDF1, testDF1) = balancedDF1.randomSplit(Array(0.8, 0.2), seed = 123)
    val Array(trainDF2, testDF2) = balancedDF2.randomSplit(Array(0.8, 0.2), seed = 123)
    val Array(trainDF3, testDF3) = balancedDF3.randomSplit(Array(0.8, 0.2), seed = 123)

    // Add index column
    val trainDFWithIndex1 = addIndexColumn(trainDF1, "cleaned_reviewText")
    val testDFWithIndex1 = addIndexColumn(testDF1, "cleaned_reviewText")
    val trainDFWithIndex2 = addIndexColumn(trainDF2, "cleaned_reviewText_stemmed")
    val testDFWithIndex2 = addIndexColumn(testDF2, "cleaned_reviewText_stemmed")
    val trainDFWithIndex3 = addIndexColumn(trainDF3, "cleaned_reviewText_lemmatized")
    val testDFWithIndex3 = addIndexColumn(testDF3, "cleaned_reviewText_lemmatized")

    // Save datasets
    saveDataset(trainDFWithIndex1, "train1.csv")
    saveDataset(testDFWithIndex1, "test1.csv")
    saveDataset(trainDFWithIndex2, "train2.csv")
    saveDataset(testDFWithIndex2, "test2.csv")
    saveDataset(trainDFWithIndex3, "train3.csv")
    saveDataset(testDFWithIndex3, "test3.csv")
    saveDataset(balancedDF1, "final1.csv")
    saveDataset(balancedDF2, "final2.csv")
    saveDataset(balancedDF3, "final3.csv")

    spark.stop()
  }

  def cleanText(df: DataFrame): DataFrame = {
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("reviewText")
      .setOutputCol("words")
      .setPattern("[^a-zA-Z\\s]")
      .setGaps(false)

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered_words")

    val stemmer = new PorterStemmer()
    val stemUDF = udf((words: Seq[String]) => words.map(word => {
      stemmer.setCurrent(word)
      stemmer.stem()
      stemmer.getCurrent
    }))

    val lemmatizeUDF = udf((text: String) => {
      val doc = new Document(text)
      doc.sentences().asScala.flatMap(_.lemmas().asScala).mkString(" ")
    })

    df.withColumn("reviewText", lower($"reviewText"))
      .transform(regexTokenizer.transform(_))
      .transform(stopWordsRemover.transform(_))
      .withColumn("cleaned_reviewText", concat_ws(" ", $"filtered_words"))
      .withColumn("cleaned_reviewText_stemmed", concat_ws(" ", stemUDF($"filtered_words")))
      .withColumn("cleaned_reviewText_lemmatized", lemmatizeUDF($"reviewText"))
      .select("overall", "cleaned_reviewText", "cleaned_reviewText_stemmed", "cleaned_reviewText_lemmatized")
  }

  def createBalancedDataset(df: DataFrame, textColumn: String): DataFrame = {
    val negativeReviews = df.filter($"overall".isin(1.0, 2.0))
    val positiveReviews = df.filter($"overall" === 5.0)
    val sampleSize = negativeReviews.count()

    val sampledPositiveReviews = positiveReviews.sample(withReplacement = false, fraction = sampleSize.toDouble / positiveReviews.count(), seed = 123)
    val balancedDF = negativeReviews.union(sampledPositiveReviews)

    balancedDF.select("overall", textColumn).orderBy(rand())
  }

  def addIndexColumn(df: DataFrame, textColumn: String): DataFrame = {
    df.withColumn("index", monotonically_increasing_id() + 1)
      .select("index", "overall", textColumn)
  }

  def saveDataset(df: DataFrame, filename: String): Unit = {
    df.coalesce(1)
      .write
      .option("header", "false")
      .option("delimiter", ",")
      .csv(filename)
  }
}