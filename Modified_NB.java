import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.Counters;

import java.io.*;
import java.io.IOException;
import java.util.*;
import java.nio.charset.StandardCharsets;


public class Modified_NB {
    public static enum Global_Counters {
        NUM_OF_REVIEWS, 
        REVIEWS_SIZE,
        POS_REVIEWS_SIZE,
        NEG_REVIEWS_SIZE,
        POS_WORDS_SIZE,
        NEG_WORDS_SIZE,
        FEATURES_SIZE,
        TRUE_POSITIVE,
        FALSE_POSITIVE,
        TRUE_NEGATIVE,
        FALSE_NEGATIVE
    }

    /* input:  <byte_offset, line_of_review>
     * output: <(word@review), 1>
     */

    public static class Map_WordCount extends Mapper<Object, Text, Text, IntWritable> {
        private Text word_review_key = new Text();
        private final static IntWritable one = new IntWritable(1);

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            context.getCounter(Global_Counters.NUM_OF_REVIEWS).increment(1);

            String line = value.toString();
            String[] columns = line.split(",");
            if (columns.length < 3) return;

            String review_id = columns[0];
            double rating = Double.parseDouble(columns[1]);
            String review_sentiment = (rating == 5.0) ? "1" : "0";
            String review_text = columns[2];
            if(review_sentiment.equals("1")) review_id += '+';

            for(String word : review_text.split(" ")) {
                if(!word.isEmpty()) {
                    word_review_key.set(word + "@" + review_id);
                    context.write(word_review_key, one);
                }
            }
        }
    }

    /* input:  <(word@review), 1>
     * output: <(word@review), word_count>
     */
    
    public static class Reduce_WordCount extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for(IntWritable value : values)
                sum += value.get();

            context.write(key, new IntWritable(sum));
        }
    }

    /* input:  <(word@review), word_count>
     * output: <review, (word=word_count)>
     */

    public static class Map_TF extends Mapper<Object, Text, Text, Text> {
        private Text review_key = new Text();
        private Text word_wordcount_value = new Text();
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String lines[] = value.toString().split("\t");
            String splitted_key[] = lines[0].toString().split("@");

            review_key.set(splitted_key[1]);
            word_wordcount_value.set(splitted_key[0] + "=" + lines[1]);

            context.write(review_key, word_wordcount_value);    
        }
    }

    
    /* input:  <review, (word=word_count)>
     * output: <(word@review), (word_count/review_length)>
     */
    
    public static class Reduce_TF extends Reducer<Text, Text, Text, Text> {
        private Text outputKey = new Text();
        private Text outputValue = new Text();
    
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int total = 0;
            Map<String, Integer> wordCounts = new HashMap<>();
    
            for (Text value : values) {
                String[] wordAndCount = value.toString().split("=");
                int count = Integer.parseInt(wordAndCount[1]);
                wordCounts.put(wordAndCount[0], count);
                total += count;
            }
    
            for (Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
                outputKey.set(entry.getKey() + "@" + key.toString());
                outputValue.set(entry.getValue() + "/" + total);
                context.write(outputKey, outputValue);
            }
        }
    }


    /* input:  <(word@review), (word_count/review_length)>
     * output: <word, (review=word_count/review_length)>
     */

    public static class Map_TFIDF extends Mapper<Object, Text, Text, Text> {
        private Text word_key = new Text();
        private Text review_wordcount_reviewsize_value = new Text();

        public void map(Object key, Text value, Context context ) throws IOException, InterruptedException {
            String[] columns = value.toString().split("\t");
            String[] splitted_key = columns[0].toString().split("@");

            word_key.set(splitted_key[0]);
            review_wordcount_reviewsize_value.set(splitted_key[1] + "=" + columns[1]);

            context.write(word_key, review_wordcount_reviewsize_value);          
        }
    }

    /* input:  <word, (review=word_count/review_length)>
     * output: <(review@word), TFIDF>
     */

    public static class Reduce_TFIDF extends Reducer<Text, Text, Text, Text> {
        private static int num_of_reviews;
        private Text outputKey = new Text();
        private Text outputValue = new Text();
    
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            num_of_reviews = Integer.parseInt(conf.get("num_of_reviews"));
        }
    
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Map<String, Double> reviewTFValues = new HashMap<>();
            int num_of_reviews_with_this_word = 0;

            for (Text value : values) {
                String[] valueParts = value.toString().split("=");
                String[] tfParts = valueParts[1].split("/");
                double wordCount = Double.parseDouble(tfParts[0]);
                double reviewLength = Double.parseDouble(tfParts[1]);
                double tf = wordCount / reviewLength;
                reviewTFValues.put(valueParts[0], tf);
                num_of_reviews_with_this_word++;
            }

            double idf = Math.log((double) num_of_reviews / num_of_reviews_with_this_word);
            
            for (Map.Entry<String, Double> entry : reviewTFValues.entrySet()) {
                double tfidf = entry.getValue() * idf;
                outputKey.set(entry.getKey() + "@" + key.toString());
                outputValue.set(String.valueOf(tfidf));
                context.write(outputKey, outputValue);
            }
        }
    }


    /* input:  <(review@word), TFIDF>
     * output: <review, (word_TFIDF)>
     */

    public static class Map_FeatSel extends Mapper<Object, Text, Text, Text> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] columns = value.toString().split("\t");
            String[] splitted_key = columns[0].toString().split("@");

            context.write(new Text(splitted_key[0]), new Text(splitted_key[1] + "_" + columns[1]));
        }
    }

    /* input:  <review, (word_TFIDF)>
     * output: <review, text>
     */

    public static class Reduce_FeatSel extends Reducer<Text, Text, Text, Text> {
        private static final double THRESHOLD_PERCENTAGE = 0.75;
    
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            context.getCounter(Global_Counters.REVIEWS_SIZE).increment(1);
            Map<String, Double> wordScores = new HashMap<>();
            String reviewText = "";

            // Track the total word count for both positive and negative reviews
            int totalPosWordCount = 0;
            int totalNegWordCount = 0;
    
            for (Text value : values) {
                String[] parts = value.toString().split("_");
                double score = Double.parseDouble(parts[1]);
                wordScores.put(parts[0], score);

                // Update total word counts
                if (score > 0) {
                    if (key.toString().endsWith("+")) {
                        totalPosWordCount++;
                    } else {
                        totalNegWordCount++;
                    }
                }
            }
    
            int numWords = wordScores.size();
            int threshold = (int) Math.ceil(numWords * THRESHOLD_PERCENTAGE);
    
            // Using a PriorityQueue to efficiently sort and trim the list of words
            PriorityQueue<Map.Entry<String, Double>> queue = new PriorityQueue<>(Comparator.comparing(Map.Entry::getValue));
            for (Map.Entry<String, Double> entry : wordScores.entrySet()) {
                queue.offer(entry);
                if (queue.size() > threshold) {
                    queue.poll();
                }
            }
    
            // Construct the final review text
            while (!queue.isEmpty()) {
                reviewText = queue.poll().getKey() + " " + reviewText;
            }
    
            // Update the counters for the number of words used in positive and negative reviews
            if (key.toString().endsWith("+")) {
                context.getCounter(Global_Counters.POS_REVIEWS_SIZE).increment(1);
                context.getCounter(Global_Counters.POS_WORDS_SIZE).increment(totalPosWordCount);
            } else {
                context.getCounter(Global_Counters.NEG_REVIEWS_SIZE).increment(1);
                context.getCounter(Global_Counters.NEG_WORDS_SIZE).increment(totalNegWordCount);
            }
            context.write(key, new Text(reviewText.trim()));
        }
    }


    /* input:  <review, text>
     * output: <word, sentiment>
     */

    public static class Map_Training extends Mapper<Object, Text, Text, Text> {
        private Text word_key = new Text();
        private Text sentiment_value = new Text();
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] line = value.toString().split("\t");
            String[] review_words = line[1].toString().split(" ");

            if(line[0].endsWith("+"))
                sentiment_value.set("POSITIVE");
            else
                sentiment_value.set("NEGATIVE");

            for(String word : review_words) {
                word_key.set(word);
                context.write(word_key, sentiment_value);
            }  
        }
    }

    /* input:  <word, sentiment>
     * output: <word, pos_wordcount@neg_wordcount>
     */

    public static class Reduce_Training extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            context.getCounter(Global_Counters.FEATURES_SIZE).increment(1); 
            int positive_counter = 0;
            int negative_counter = 0;
            // for each word, count the occurrences in reviews with positive/negative sentiment
            for(Text value : values) {
                String sentiment = value.toString();
                if(sentiment.equals("POSITIVE"))
                    positive_counter++;
                else
                    negative_counter++;
            }
            context.write(key, new Text(String.valueOf(positive_counter) + "@" + String.valueOf(negative_counter)));    
        }
    }

    /* input: <byte_offset, line_of_review>
     * output: <review@review_text, sentiment>
     */

    public static class Map_Testing extends Mapper<Object, Text, Text, Text> {
        private int features_size, pos_words_size, neg_words_size, reviews_size, pos_reviews_size, neg_reviews_size;
        private double pos_class_probability, neg_class_probability;
        private HashMap<String, Double> pos_word_probabilities = new HashMap<>();
        private HashMap<String, Double> neg_word_probabilities = new HashMap<>();
    
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            loadModelAndCalculateClassProbabilities(conf);
        }
    
        private void loadModelAndCalculateClassProbabilities(Configuration conf) throws IOException {
            features_size = Integer.parseInt(conf.get("features_size"));
            pos_words_size = Integer.parseInt(conf.get("pos_words_size"));
            neg_words_size = Integer.parseInt(conf.get("neg_words_size"));
            // Assuming model is stored in HDFS in 'training' directory
            Path trainingModelPath = new Path("training");
            FileSystem fs = FileSystem.get(conf);
            FileStatus[] fileStatuses = fs.listStatus(trainingModelPath);
    
            for (FileStatus status : fileStatuses) {
                if (status.isFile()) {
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(status.getPath())))) {
                        String line;
                        while ((line = br.readLine()) != null) {
                            String[] parts = line.split("\t");
                            String word = parts[0];
                            String[] counts = parts[1].split("@");
                            int pos_count = Integer.parseInt(counts[0]);
                            int neg_count = Integer.parseInt(counts[1]);
    
                            double pos_prob = (double) (pos_count + 1) / (pos_words_size + features_size);
                            double neg_prob = (double) (neg_count + 1) / (neg_words_size + features_size);
    
                            pos_word_probabilities.put(word, pos_prob);
                            neg_word_probabilities.put(word, neg_prob);
                        }
                    }
                }
            }
            
            // Calculate class probabilities
            reviews_size = Integer.parseInt(conf.get("reviews_size"));
            pos_reviews_size = Integer.parseInt(conf.get("pos_reviews_size"));
            neg_reviews_size = Integer.parseInt(conf.get("neg_reviews_size"));
    
            pos_class_probability = (double) pos_reviews_size / reviews_size;
            neg_class_probability = (double) neg_reviews_size / reviews_size;
        }
    
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] columns = line.split(",");
            if (columns.length < 3) return;
         
            String review_id = columns[0];
            double rating = Double.parseDouble(columns[1]);
            String actual_sentiment = (rating == 5.0) ? "1" : "0";
            String review_text = columns[2];
    
            double pos_probability = pos_class_probability;
            double neg_probability = neg_class_probability;
            for (String word : review_text.split(" ")) {
                if(!word.isEmpty()) {
                    pos_probability *= pos_word_probabilities.getOrDefault(word, 1.0 / (pos_words_size + features_size));
                    neg_probability *= neg_word_probabilities.getOrDefault(word, 1.0 / (neg_words_size + features_size));
                }
                pos_probability *= pos_word_probabilities.getOrDefault(word, 1.0 / (pos_words_size + features_size));
                neg_probability *= neg_word_probabilities.getOrDefault(word, 1.0 / (neg_words_size + features_size));
            }     
            String predicted_sentiment = pos_probability > neg_probability ? "POSITIVE" : "NEGATIVE";
            if ("POSITIVE".equals(predicted_sentiment) && "1".equals(actual_sentiment)) {
                context.getCounter(Global_Counters.TRUE_POSITIVE).increment(1);
            } else if ("POSITIVE".equals(predicted_sentiment) && "0".equals(actual_sentiment)) {
                context.getCounter(Global_Counters.FALSE_POSITIVE).increment(1);
            } else if ("NEGATIVE".equals(predicted_sentiment) && "0".equals(actual_sentiment)) {
                context.getCounter(Global_Counters.TRUE_NEGATIVE).increment(1);
            } else if ("NEGATIVE".equals(predicted_sentiment) && "1".equals(actual_sentiment)) {
                context.getCounter(Global_Counters.FALSE_NEGATIVE).increment(1);
            }
            context.write(new Text(review_id + "@" + review_text), new Text(predicted_sentiment));
        }
    }

    public static void main(String[] args) throws Exception {
        // paths to directories were input, inbetween and final job outputs are stored
        Path trainingInputPath = new Path(args[0]);
        Path testingInputPath = new Path(args[1]);
        Path outputPath = new Path(args[2]);

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        Path wordcount_dir = new Path(outputPath, "wordcount");
        Path tf_dir = new Path(outputPath, "tf");
        Path tfidf_dir = new Path(outputPath, "tfidf");
        Path features_dir = new Path(outputPath, "features");
        Path training_dir = new Path(outputPath, "training");
        Path output_dir = new Path(outputPath, "output");

        // Delete existing directories if they exist
        if(fs.exists(wordcount_dir)) {
            fs.delete(wordcount_dir, true); // true for recursive delete
        }
        if(fs.exists(tf_dir)) {
            fs.delete(tf_dir, true);
        }
        if(fs.exists(tfidf_dir)) {
            fs.delete(tfidf_dir, true);
        }
        if(fs.exists(features_dir)) {
            fs.delete(features_dir, true);
        }
        if(fs.exists(training_dir)) {
            fs.delete(training_dir, true);
        }
        if(fs.exists(output_dir)) {
            fs.delete(output_dir, true);
        }

        long start_time = System.nanoTime();

        Job wordcount_job = Job.getInstance(conf, "Word Count");
        wordcount_job.setJarByClass(Modified_NB.class);
        wordcount_job.setMapperClass(Map_WordCount.class);
        wordcount_job.setCombinerClass(Reduce_WordCount.class);
        wordcount_job.setReducerClass(Reduce_WordCount.class);
        wordcount_job.setNumReduceTasks(3);
        wordcount_job.setMapOutputKeyClass(Text.class);
        wordcount_job.setMapOutputValueClass(IntWritable.class);
        wordcount_job.setOutputKeyClass(Text.class);
        wordcount_job.setOutputValueClass(IntWritable.class);
        TextInputFormat.addInputPath(wordcount_job, trainingInputPath);
        TextOutputFormat.setOutputPath(wordcount_job, wordcount_dir);
        wordcount_job.waitForCompletion(true);

        // Counting the total number of reviews from the training data in order to calculate the TFIDF score of each feature
        int num_of_reviews = Math.toIntExact(wordcount_job.getCounters().findCounter(Global_Counters.NUM_OF_REVIEWS).getValue());
        conf.set("num_of_reviews", String.valueOf(num_of_reviews));

        Job tf_job = Job.getInstance(conf, "TF");
        tf_job.setJarByClass(Modified_NB.class);
        tf_job.setMapperClass(Map_TF.class);
        tf_job.setReducerClass(Reduce_TF.class);
        tf_job.setNumReduceTasks(3);
        tf_job.setMapOutputKeyClass(Text.class);
        tf_job.setMapOutputValueClass(Text.class);
        tf_job.setOutputKeyClass(Text.class);
        tf_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(tf_job, wordcount_dir);
        FileOutputFormat.setOutputPath(tf_job, tf_dir);
        tf_job.waitForCompletion(true);

        Job tfidf_job = Job.getInstance(conf, "TFIDF");
        tfidf_job.setJarByClass(Modified_NB.class);
        tfidf_job.setMapperClass(Map_TFIDF.class);
        tfidf_job.setReducerClass(Reduce_TFIDF.class);
        tfidf_job.setNumReduceTasks(3);
        tfidf_job.setMapOutputKeyClass(Text.class);
        tfidf_job.setMapOutputValueClass(Text.class);
        tfidf_job.setOutputKeyClass(Text.class);
        tfidf_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(tfidf_job, tf_dir);
        FileOutputFormat.setOutputPath(tfidf_job, tfidf_dir);
        tfidf_job.waitForCompletion(true);

        Job feature_selection_job = Job.getInstance(conf, "Feature Selection");
        feature_selection_job.setJarByClass(Modified_NB.class);
        feature_selection_job.setMapperClass(Map_FeatSel.class);
        feature_selection_job.setReducerClass(Reduce_FeatSel.class);
        feature_selection_job.setNumReduceTasks(3);
        feature_selection_job.setMapOutputKeyClass(Text.class);
        feature_selection_job.setMapOutputValueClass(Text.class);
        feature_selection_job.setOutputKeyClass(Text.class);
        feature_selection_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(feature_selection_job, tfidf_dir);
        FileOutputFormat.setOutputPath(feature_selection_job, features_dir);
        feature_selection_job.waitForCompletion(true);

        int reviews_size = Math.toIntExact(feature_selection_job.getCounters().findCounter(Global_Counters.REVIEWS_SIZE).getValue());
        conf.set("reviews_size", String.valueOf(reviews_size));
        int pos_reviews_size = Math.toIntExact(feature_selection_job.getCounters().findCounter(Global_Counters.POS_REVIEWS_SIZE).getValue());
        conf.set("pos_reviews_size", String.valueOf(pos_reviews_size));
        int neg_reviews_size = Math.toIntExact(feature_selection_job.getCounters().findCounter(Global_Counters.NEG_REVIEWS_SIZE).getValue());
        conf.set("neg_reviews_size", String.valueOf(neg_reviews_size));
        int pos_words_size = Math.toIntExact(feature_selection_job.getCounters().findCounter(Global_Counters.POS_WORDS_SIZE).getValue());
        conf.set("pos_words_size", String.valueOf(pos_words_size));
        int neg_words_size = Math.toIntExact(feature_selection_job.getCounters().findCounter(Global_Counters.NEG_WORDS_SIZE).getValue());
        conf.set("neg_words_size", String.valueOf(neg_words_size));

        Job training_job = Job.getInstance(conf, "Training");
        training_job.setJarByClass(Modified_NB.class);
        training_job.setMapperClass(Map_Training.class);
        training_job.setReducerClass(Reduce_Training.class);  
        training_job.setNumReduceTasks(3);   
        training_job.setMapOutputKeyClass(Text.class);
        training_job.setMapOutputValueClass(Text.class);
        training_job.setOutputKeyClass(Text.class);
        training_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(training_job, features_dir);
        FileOutputFormat.setOutputPath(training_job, training_dir);
        training_job.waitForCompletion(true);

        int features_size = Math.toIntExact(training_job.getCounters().findCounter(Global_Counters.FEATURES_SIZE).getValue());
        conf.set("features_size", String.valueOf(features_size));

        Job testing_job = Job.getInstance(conf, "Testing");
        testing_job.setJarByClass(Modified_NB.class);
        testing_job.setMapperClass(Map_Testing.class);  
        testing_job.setMapOutputKeyClass(Text.class);
        testing_job.setMapOutputValueClass(Text.class);
        testing_job.setOutputKeyClass(Text.class);
        testing_job.setOutputValueClass(Text.class);
        TextInputFormat.addInputPath(testing_job, testingInputPath);
        TextOutputFormat.setOutputPath(testing_job, output_dir);
        testing_job.waitForCompletion(true);

        System.out.println("EXECUTION DURATION: " + (System.nanoTime() - start_time) / 1000000000F + " seconds");

        int tp = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.TRUE_POSITIVE).getValue());
        int fp = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.FALSE_POSITIVE).getValue());
        int tn = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.TRUE_NEGATIVE).getValue());
        int fn = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.FALSE_NEGATIVE).getValue());

        System.out.println("\nCONFUSION MATRIX:");
        System.out.printf("%-10s %-10s \n", tp, fp);
        System.out.printf("%-10s %-10s \n\n", fn, tn);
        System.out.printf("%-25s %-10s \n", "ACCURACY: ", ((double) (tp + tn)) / (tp + fp + tn + fn));
        System.out.printf("%-25s %-10s \n", "PRECISION: ", ((double) tp) / (tp + fp));
        System.out.printf("%-25s %-10s \n", "RECALL: ", ((double) tp) / (tp + fn));
        System.out.printf("%-25s %-10s \n", "F1-SCORE: ", ((double) (2 * tp)) / (2 * tp + fp + fn));
    }
}
