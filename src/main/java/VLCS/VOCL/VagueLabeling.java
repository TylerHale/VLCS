package VLCS.VOCL;

import weka.core.Instances;

/**
 * Interface for VagueLabeling methods.
 */
public abstract class VagueLabeling {
    protected float alpha = 0.3f;

    public abstract int[] label(Instances chunk) throws Exception;
}
