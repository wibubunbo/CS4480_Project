name := "DataCleaning"
version := "0.1"
scalaVersion := "2.12.15"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.0",
  "org.apache.spark" %% "spark-sql" % "3.2.0",
  "org.apache.spark" %% "spark-mllib" % "3.2.0",
  "org.apache.lucene" % "lucene-analyzers-common" % "8.8.2",
  "edu.stanford.nlp" % "stanford-corenlp" % "4.2.0",
  "edu.stanford.nlp" % "stanford-corenlp" % "4.2.0" classifier "models"
)