package VLCS.VOCL;

import weka.core.Instance;
import weka.core.Instances;

import java.util.Queue;

/**
 * Global Weighting Formula method from paper (Eq 2.)
 */
public class GlobalWeighting {

    public float[] getWeights(Instances chunk, OneClassClassifierEnsemble ensemble) throws Exception {

        float[] weights = new float[chunk.size()];

        System.err.println("GLOBAL_WEIGHTING");

        Queue<MOAOneClassClassifier> classifiers = ensemble.getClassifiers();

        for (int i = 0; i < weights.length; i++) {

            float percent_predicted_positive = 0.0f;
            Instance instance = chunk.instance(i);

            // Get the percentage of classifiers in the ensemble that predicted
            // the instance of being positively labelled
            for (MOAOneClassClassifier classifier : classifiers) {
                double index = classifier.classifyInstance(instance);
                // If predict's as positive (part of class)
                if ((int) index == Integer.parseInt(classifier.getTargetClassLabel())) {
                    percent_predicted_positive += 1.0f;
                }
            }

            weights[i] = percent_predicted_positive;

            if (classifiers.size() > 0) {
                weights[i] /= classifiers.size();
            }
            System.err.println("% Predict_positive: " + weights[i]);
        }

        return weights;
    }
}
