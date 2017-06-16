package VLCS.VOCL;

import weka.classifiers.meta.OneClassClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Arrays;
import java.util.Collections;

public class LocalWeighting {
    private int folds;

    public LocalWeighting(int folds) {
        if (folds < 3) {
            throw new IllegalArgumentException("Number of folds has to be at least 3.");
        }
        this.folds = folds;
    }

    /**
     * Local Weighting Pseudo code method from paper (Figure 6.)
     *
     * @param chunk         The dataset to train and test from
     * @param labels        Identifies which instances in the chunk are positively labeled or unlabeled
     * @param attribute_idx The attribute to predict from
     * @param class_idx     The class to try classify
     * @return The weights of each instance in the chunk.
     * @throws Exception
     */
    public float[] getWeights(Instances chunk, int[] labels, int attribute_idx, int class_idx) throws Exception {

        assert chunk.size() % folds == 0 : "chunk size has to be multiple of num folds.";
        assert labels.length == chunk.size() : "number of labels does not match chunk size.";

        OneClassClassifier[] classifiers = new OneClassClassifier[folds];

        float[] weights = new float[chunk.size()];
        float min = 0.f, max = 2.f;

        System.out.println("-----------------");

        int[] was_pos_predicted = new int[weights.length];

        for (int i = 0; i < folds; i++) {
            classifiers[i] = new OneClassClassifier();
            // Specify attribute interest
            classifiers[i].setTargetClassLabel(Integer.toString(class_idx));

            Instances training_set = chunk.trainCV(folds, i);
            Instances testing_set = chunk.testCV(folds, i);

            // Train the one-class classifiers to the specified class-idx with the training data
            classifiers[i].buildClassifier(training_set);

            for (int p = i * testing_set.size(); p < (i + 1) * testing_set.size(); p++) {
                Instance instance = testing_set.instance(p - (i * testing_set.size()));
                double index = classifiers[i].classifyInstance(instance);
                // Update for positive samples
                if (labels[p] == 1) {
                    // If Classifies p as positive then set weight to 1.0 else 0.5;
                    weights[p] = ((int) index == class_idx) ? 1.0f : 0.5f;
                    was_pos_predicted[p] = ((int) index == class_idx) ? 1 : 0; // TODO remove
                    // Update for unlabelled samples that are Classified as positive
                } else if (labels[p] == 0 && (int) index == class_idx) {
                    // Predict unlabelled instance using all <= i OneClassClassifiers to calculate weight
                    float weight = 0.f;
                    for (int f = 0; f < i; f++) {
                        double sub_index = classifiers[f].classifyInstance(instance);
                        // Increase weight if positive
                        weight += ((int) sub_index == class_idx) ? 1.0f : 0.0f;
                    }
                    was_pos_predicted[p] = 1; // TODO remove
                    weights[p] = (weight / folds) + 1;
                    // Normalize weight values to 0-1 for unlabelled set (USi)
                    weights[p] = (weights[p] - min) / (max - min);
                    assert weights[p] >= 0 && weights[p] <= 1 : "normalized weight must be between 0-1.";
                }

                System.err.println("Is pos: " + labels[p] + " Was_pos_pred: " + was_pos_predicted[p] + " WLx[p]: " + weights[p]);
            }
        }

        return weights;
    }
}
