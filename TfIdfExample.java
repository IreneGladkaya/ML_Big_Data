import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.linalg.Vector;


public class TfIdfExample {
        public static void main(String[] args){
            SparkSession spark = SparkSession.builder().config("spark.master", "local").appName("Simple Application").getOrCreate();

            String path = "/Users/irinabezgina/Downloads/Chehov_Anton__Tri_sestry_www.Readli.Net_5870.txt";

            List<Row> data1 = Arrays.asList(
                    RowFactory.create(0.0, "Hi I heard about Spark"),
                    RowFactory.create(1.0, "I wish Java could use case classes"),
                    RowFactory.create(2.0, "Logistic regression models are neat"),
                    RowFactory.create(3.0, "Logistic regression models are neat"),
                    RowFactory.create(4.0, "Logistic regression models are neat"),
                    RowFactory.create(5.0, "Logistic regression models are neat")
            );
            StructType schema = new StructType(new StructField[]{
                    new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                    new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
            });
            Dataset<Row> sentenceData = spark.createDataFrame(data1, schema);

            Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
            Dataset<Row> wordsData = tokenizer.transform(sentenceData);

            int numFeatures = 20;
            HashingTF hashingTF = new HashingTF()
                    .setInputCol("words")
                    .setOutputCol("rawFeatures")
                    .setNumFeatures(numFeatures);

            Dataset<Row> featurizedData = hashingTF.transform(wordsData);

            IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
            IDFModel idfModel = idf.fit(featurizedData);

            Dataset<Row> rescaledData = idfModel.transform(featurizedData);
            rescaledData.select("label", "features").show(3,200);


// Cluster the data into two classes using KMeans
            KMeans kmeans = new KMeans().setK(2).setSeed(1L);
            KMeansModel model = kmeans.fit(rescaledData);

// Make predictions
            Dataset<Row> predictions = model.transform(rescaledData);

            predictions.show();

// Evaluate clustering by computing Silhouette score
            ClusteringEvaluator evaluator = new ClusteringEvaluator();

            double silhouette = evaluator.evaluate(predictions);
            System.out.println("Silhouette with squared euclidean distance = " + silhouette);

// Shows the result.
            Vector[] centers = model.clusterCenters();
            System.out.println("Cluster Centers: ");
            for (Vector center: centers) {
                System.out.println(center);
            }

// Save and load model
            spark.stop();
        }
    }
