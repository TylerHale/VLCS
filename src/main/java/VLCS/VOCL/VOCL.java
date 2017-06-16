package VLCS.VOCL;

import java.util.*;

import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class VOCL {

    // Paper said RandomVagueLabeling was more accurate however Cluster based
    // vague labeling seemed more interesting so we implemented that one.
    public enum VagueLabelMethod {
        RANDOM /* TODO */, CLUSTER
    }

    private VagueLabelMethod label_method;
    private final int num_clusters = 20; // Specified in paper
    private int k = 10; // Specified in paper
    private int class_idx;
    private int attr_idx;

    private VagueLabeling vague_clustering = new ClusterVagueLabeling();

    private LocalWeighting local;
    private GlobalWeighting global;
    private OneClassClassifierEnsemble ensemble = new OneClassClassifierEnsemble(k);
    private static Remove filter = new Remove();

    public VOCL(VagueLabelMethod label_method, int class_idx, int attr_idx) {
        this.label_method = label_method;
        this.local = new LocalWeighting(10); // Specify number of folds
        this.global = new GlobalWeighting();
        this.class_idx = class_idx;
        this.attr_idx = attr_idx; // TODO make dynamic for probability shifting
    }

    /**
     * This is the main VOCL method. This will apply vague one-class learning on the input stream.
     *
     * @param stream     The stream of instances
     * @param chunk_size The number of instances per chunk
     * @throws Exception
     */
    public void labelStream(Instances stream, final int chunk_size) throws Exception {
        assert chunk_size >= num_clusters : "chunk size must be >= number of clusters.";

        Instances chunk = new Instances(stream);
        chunk.clear();

        System.out.println("Finished clearing");

        for (Instance instance : stream) {
            chunk.add(instance);
            if (chunk.size() == chunk_size) {
                double predicted_accuracy = processChunk(chunk);
                System.out.println("Predicted_accuracy: " + predicted_accuracy);
                chunk.clear();
            }
        }

        // Does not process partial chunks
    }

    /**
     * Main Pseudo code method from paper (Figure 5.)
     *
     * @param chunk The chunk to test for the previous ensemble and the chunk to train for the next classifier
     * @return The predict chunk accuracy
     * @throws Exception
     */
    private double processChunk(Instances chunk) throws Exception {

        double predicted_accuracy = 0.f;

        int[] PSi = vague_clustering.label(chunk);

        // Create copy: Filter chunk by attribute & specify chunk class index
        Instances attr_chunk = filterByAttribute(chunk, attr_idx);
        attr_chunk.setClassIndex(attr_chunk.numAttributes() - 1);

        // We can classify as ensemble contains trained classifiers
        if (ensemble.size() > 0) {
            predicted_accuracy = evaluateChunkAccuracy(attr_chunk);
        }

        float[] WLx = local.getWeights(attr_chunk, PSi, attr_idx, class_idx);
        float[] WGx = global.getWeights(attr_chunk, ensemble);
        float[] Wx = calculateUnifiedWeights(WLx, WGx);

        MOAOneClassClassifier Li = trainNewClassifier(attr_chunk, Wx);

        if (ensemble.size() > 0) {
            assert ensemble.size() != k : "ensemble size must be == k - 1";
            float[] Gl = weightClassifiers(Li, Wx, attr_chunk, PSi);

            ensemble.updateClassifierWeights(Gl);
        }

        // Append new classifier to ensemble for make it k classifiers
        ensemble.pushClassifier(Li);

        // Shift out oldest classifier
        if (ensemble.size() == k) ensemble.popLastClassifier();

        return predicted_accuracy;
    }

    /**
     * Returns the ensemble prediction from a given instance.
     *
     * @param test The instance to predict.
     * @return The Prediction containing the actual and predicted results.
     * @throws Exception
     */
    public Prediction getPrediction(Instance test) throws Exception {

        double actual = test.classValue();
        double[] dist = ensemble.getVotesForInstance(test);
        if (test.classAttribute().isNominal()) {
            return new NominalPrediction(actual, dist, test.weight());
        } else {
            return new NumericPrediction(actual, dist[0], test.weight());
        }
    }

    /**
     * For each instance it checks whether it predicts it should be in the class or not and then
     * calculates its overall accuracy for the classifier on the given chunk.
     *
     * @param chunk The chunk to predict
     * @return The overall accuracy
     * @throws Exception
     */
    private double evaluateChunkAccuracy(Instances chunk) throws Exception {

        int correct_predictions = 0;
        for (Instance instance : chunk) {
            Prediction pred = getPrediction(instance);

            // If it's not part of class then ensure it predicted it wasn't.
            if ((int) pred.actual() != class_idx && (int) pred.predicted() != class_idx) {
                correct_predictions++;
            }

            // If its part of the class then sure it predicted it was.
            if ((int) pred.actual() == class_idx && (int) pred.predicted() == class_idx) {
                correct_predictions++;
            }
        }

        return (double) correct_predictions / (double) chunk.size();
    }

    private float[] weightClassifiers(MOAOneClassClassifier Li, float[] Wx, Instances chunk, int[] labels) throws Exception {

        float[] Gl = new float[ensemble.size()];

        Gl[ensemble.size() - 1] = pairWiseAgreement(Li, Li, chunk, labels);
        assert Float.compare(Gl[ensemble.size() - 1], 1.0f) == 0 : "pair-wise agreement against itself must be 1.";

        int i = 0;
        // Iterate from last recently used to most recently used (excludes Li since it hasn't been appended yet)
        Queue<MOAOneClassClassifier> classifiers = ensemble.getClassifiers();
        for (MOAOneClassClassifier classifier : classifiers) {
            Gl[i++] = pairWiseAgreement(Li, classifier, chunk, labels);
        }

        return Gl;
    }

    /**
     * Compares oj classifier with o1 in order to create weight for oj classifier.
     *
     * @param ol     The classifier to compare against
     * @param oj     The classifier to weight
     * @param chunk  The dataset
     * @param labels The labels identifying whether and instance is positively labeled or unlabeled.
     * @return The oj classifier weight
     * @throws Exception
     */
    private float pairWiseAgreement(MOAOneClassClassifier ol, MOAOneClassClassifier oj, Instances chunk, int[] labels) throws Exception {

        int unlabeled_set_size = 0;
        float weight = 0.f;

        for (int i = 0; i < chunk.size(); i++) {
            // If unlabeled
            if (labels[i] == 0) {
                Instance instance = chunk.instance(i);
                // If classifiers predict same value increase weight
                if (Double.compare(ol.classifyInstance(instance), oj.classifyInstance(instance)) == 0) {
                    weight += 1.0f;
                }
                unlabeled_set_size++;
            }
        }

        return weight / unlabeled_set_size;
    }

    /**
     * Calculates unified weights using both Local and Global weights
     *
     * @param WLx The array of local weights
     * @param WGx The array of global weights
     * @return The array of unified weights
     */
    private float[] calculateUnifiedWeights(float[] WLx, float[] WGx) {

        assert WLx.length == WGx.length : "Weights have different lengths.";

        // Laplace smoothing parameters
        float a = 0.5f, b = 0.5f;

        float[] Wx = new float[WLx.length];

        System.err.println("UNIFIED_WEIGHTING");

        for (int i = 0; i < Wx.length; i++) {
            if (WLx[i] + WGx[i] > 0) {
                Wx[i] = (WLx[i] + WGx[i] + a) / (Math.abs(WLx[i] - WGx[i]) + b);
                assert Wx[i] >= 1 && Wx[i] <= 5 : "Unified weight expected to be between 1-5.";
            } else {
                Wx[i] = 0.f;
            }
            System.err.println("LocalW: " + WLx[i] + " GlobalW: " + WGx[i] + " UnifiedW: " + Wx[i]);
        }

        return Wx;
    }

    /**
     * Trains a new classifier using unified weights.
     * <p>
     * NOTE: Had to use weka's buildClassifier instead of trainOnInstance so that I could
     * <p>
     * Trains a new classifier given the unified weights.
     *
     * @param chunk           The testing data set
     * @param unified_weights The weights to train it by
     * @return The trained classifier
     * @throws Exception
     */
    private MOAOneClassClassifier trainNewClassifier(Instances chunk, float[] unified_weights) throws Exception {

        Instances new_chunk = new Instances(chunk);

        // Create new classifier to train
        MOAOneClassClassifier new_classifier = new MOAOneClassClassifier();
        new_classifier.setTargetClassLabel(Integer.toString(class_idx));

        // Update chunk with new weights
        for (int i = 0; i < new_chunk.size(); i++) {
            Instance instance = new_chunk.instance(i);
            // TODO Rejection sampling
            instance.setWeight(unified_weights[i]);
            new_chunk.set(i, instance);
        }

        // Train classifier with new weighted chunk
        // OneClassClassifier requires training this way opposed to trainOnInstance
        new_classifier.buildClassifier(new_chunk);

        return new_classifier;
    }

    /**
     * Filters out unwanted attributes from each instance.
     *
     * @param data          The training/testing fold to filter.
     * @param attribute_idx The only attribute you want in each instance.
     * @return The filtered chunk.
     * @throws Exception
     */
    public static Instances filterByAttribute(Instances data, int attribute_idx) throws Exception {
        filter.setAttributeIndices(Integer.toString(attribute_idx)); // attr0, attr1, class (attr0 = index 2 NOT 0!)
        filter.setInputFormat(data);
        Instances new_data = Filter.useFilter(data, filter);
        new_data.setClassIndex(new_data.numAttributes() - 1);
        return new_data;
    }
}
