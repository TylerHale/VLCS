package VLCS.VOCL;

import moa.classifiers.meta.WEKAClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.OneClassClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This class allows us to use the OneClassClassifier from Weka in Moa.
 * Inherits the WEKAClassifier from Moa to allow us to access based members
 * such that we can replicate functionality. Implement Moa Classifier so we can
 * use it with other Moa Ensembles.
 */
public class MOAOneClassClassifier extends WEKAClassifier implements Classifier {

    public MOAOneClassClassifier() {
        this.classifier = new OneClassClassifier();
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        this.classifier.buildClassifier(instances);
    }


    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return this.classifier.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return this.classifier.distributionForInstance(instance);
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] votes = new double[inst.numClasses()];
        try {
            votes = this.classifier.distributionForInstance(inst);
        } catch (Exception var4) {
            System.err.println(var4.getMessage());
        }
        return votes;
    }

    @Override
    public Capabilities getCapabilities() {
        return this.classifier.getCapabilities();
    }

    /**
     * WEKA specific method that's unimplemented in MOA.
     *
     * @param label
     */
    public void setTargetClassLabel(String label) {
        ((OneClassClassifier) this.classifier).setTargetClassLabel(label);
    }

    /**
     * WEKA specific method that's unimplemented in MOA.
     *
     * @return
     */
    public String getTargetClassLabel() {
        return ((OneClassClassifier) this.classifier).getTargetClassLabel();
    }
}
