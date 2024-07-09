import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Stemmer}

object DataCleaning {
  def main(args: Array[String]): Unit = {
    // Create SparkSession
    val spark = SparkSession.builder()
      .appName("Amazon Review Sentiment Analysis")
      .getOrCreate()

    // Read the JSON files
    val reviewDF = spark.read.json("path/to/Cell_Phones_and_Accessories.json")
    val metaDF = spark.read.json("path/to/meta_Cell_Phones_and_Accessories.json")

    // Select and clean the columns we need
    val cleanedDF = reviewDF.select(
      col("overall"),
      col("reviewText")
    ).filter(col("reviewText").isNotNull)

    // Define cleaning function
    def cleanText(text: String): String = {
      text.replaceAll("<[^>]*>", "") // Remove HTML tags
        .replaceAll("(https?:\\/\\/www\\.|https?:\\/\\/|www\\.)[\\w\\-]+([\\.\\/\\-\\w]*)*\\S+", "") // Remove URLs
        .replaceAll("\\S*(#|@|&)\\S*", "") // Remove usernames, hashtags, ampersands
        .replaceAll("\\d+", "") // Remove numbers
        .replaceAll("[^a-zA-Z ]", "") // Remove special characters, punctuations
        .toLowerCase // Convert to lowercase
    }

    // Register the UDF
    val cleanTextUDF = udf(cleanText _)

    // Apply cleaning function
    val cleanedTextDF = cleanedDF.withColumn("cleaned_reviewText", cleanTextUDF(col("reviewText")))

    // Tokenize the cleaned text
    val tokenizer = new RegexTokenizer()
      .setInputCol("cleaned_reviewText")
      .setOutputCol("words")
      .setPattern("\\W+")

    val tokenizedDF = tokenizer.transform(cleanedTextDF)

    // Remove stop words
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered_words")

    val removedStopWordsDF = remover.transform(tokenizedDF)

    // Apply stemming
    val stemmer = new Stemmer()
      .setInputCol("filtered_words")
      .setOutputCol("stemmed_words")

    val stemmedDF = stemmer.transform(removedStopWordsDF)

    // Join words back into a single string
    val finalDF = stemmedDF.withColumn("final_text", concat_ws(" ", col("stemmed_words")))

    // Filter out reviews with only one word
    val multiWordDF = finalDF.filter(size(col("stemmed_words")) > 1)

    // Prepare data for sentiment analysis
    val sentimentDF = multiWordDF.withColumn("sentiment", when(col("overall") === 5.0, 1).otherwise(0))

    // Split into train and test sets
    val Array(trainDF, testDF) = sentimentDF.randomSplit(Array(0.8, 0.2), seed = 42)

    // Save train and test sets
    trainDF.write.csv("path/to/repo/train_data.csv")
    testDF.write.csv("/path/to/repo/test_data.csv")

    // Stop SparkSession
    spark.stop()
  }
}