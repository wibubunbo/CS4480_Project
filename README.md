# Amazon Review Sentiment Analysis

This repository contains the sentiment analysis project for Amazon Review Data in the category of Cell Phones and Accessories. This project is part of the coursework for CS4480 - Data-intensive Computing at City University of Hong Kong.

## Prerequisites

- Java 8 or higher
- Scala 2.12.15
- Apache Spark 3.2.0
- Apache Hadoop 3.x
- SBT (Scala Build Tool)
- Python 3.7+
- Miniconda/Conda/Virtualenv

## Getting Started

To use this project, first clone the repository and install the required Python dependencies:

```
git clone https://github.com/wibubunbo/CS4480_Project.git
cd ./CS4480_Project
pip install -r requirements.txt
```

## Data Acquisition

To acquire the necessary data, use the following commands:

```
!wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Cell_Phones_and_Accessories.json.gz
!wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Cell_Phones_and_Accessories.json.gz
```

After downloading, unzip the files to proceed with data processing and cleaning.

## Data Processing and Cleaning

We use Apache Spark with Scala for data cleaning and preprocessing. The main cleaning script is `DataCleaning.scala`.

To compile and run the data cleaning process:

```
sbt compile
sbt "runMain DataCleaning"
```

## Hadoop File System Preparation

Create the necessary directories in HDFS, make sure you already started all Hadoop daemons, the namenode, datanodes by `start-all.sh` first.

```
hdfs dfs -mkdir /Amazon_Project
hdfs dfs -mkdir /Amazon_Project_Stem
hdfs dfs -mkdir /Amazon_Project_Lemma

hdfs dfs -mkdir /Amazon_Project/Input
hdfs dfs -mkdir /Amazon_Project_Stem/Input
hdfs dfs -mkdir /Amazon_Project_Lemma/Input

hdfs dfs -mkdir /Amazon_Project/Output
hdfs dfs -mkdir /Amazon_Project_Stem/Output
hdfs dfs -mkdir /Amazon_Project_Lemma/Output
```

## Hadoop MapReduce Implementation

#### Preparing the Data

Put the training and test datasets into HDFS using the following commands:

For the dataset without stemming and lemmatization:

```
hdfs dfs -put ./train1.csv /Amazon_Project/Input
hdfs dfs -put ./test1.csv /Amazon_Project/Input
```

For the dataset with stemming:

```
hdfs dfs -put ./train2.csv /Amazon_Project_Stem/Input
hdfs dfs -put ./test2.csv /Amazon_Project_Stem/Input
```

For the dataset with lemmatization:

```
hdfs dfs -put ./train3.csv /Amazon_Project_Lemma/Input
hdfs dfs -put ./test3.csv /Amazon_Project_Lemma/Input
```

#### Running the Naive Bayes Model

`Modified_NB.java` contains the implementation of the Naive Bayes algorithm. Compile and run the code using the following commands:

```
javac -classpath $(hadoop classpath) -d Modified_NB_classes Modified_NB.java
jar -cvf Modified_NB.jar -C Modified_NB_classes/ .
hadoop jar Modified_NB.jar Modified_NB /Amazon_Project/Input/train1.csv /Amazon_Project/Input/test1.csv /Amazon_Project/Output
```

Repeat the `hadoop jar` command as necessary for the other datasets.

#### Retrieving Output Data

After completing the MapReduce jobs, merge and retrieve the output job files with the following commands:

```
hdfs dfs -getmerge /Amazon_Project/Output/wordcount ./wordcount.txt
hdfs dfs -getmerge /Amazon_Project/Output/tfidf ./tfidf.txt
hdfs dfs -getmerge /Amazon_Project/Output/features ./features.txt
```

## Visualization

To visualize the results after the Hadoop MapReduce process, use the `visualization_result.ipynb` Jupyter Notebook.
