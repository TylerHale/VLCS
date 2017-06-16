package VLCS.VOCL;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

import java.util.*;
import java.util.Map.Entry;
import java.util.AbstractMap.SimpleEntry;

public class ClusterVagueLabeling extends VagueLabeling {

    private final int k = 20; // Recommended in paper

    private int[] cluster_sizes = new int[k]; // Allocate once

    // TODO replace with automatically sorting data structure
    private List<Entry<Float, Integer>> sorted_clusters = new ArrayList<>(); // Allocate once

    /**
     * Clusters the data into k clusters
     *
     * @param chunk The data to cluster
     * @return The cluster ids from 0-(k-1)
     * @throws Exception
     */
    private int[] getClusterIDs(Instances chunk) throws Exception {
        SimpleKMeans k_means = new SimpleKMeans();
        k_means.setPreserveInstancesOrder(true);
        k_means.setNumClusters(k);
        k_means.buildClusterer(chunk);
        return k_means.getAssignments();
    }

    /**
     * Return a labeled array with 1 denoting a positive sample and a 0 denoting an unlabeled sample.
     *
     * @param chunk The data to label
     * @return The positively labeled data
     * @throws Exception
     */
    @Override
    public int[] label(Instances chunk) throws Exception {

        assert chunk.size() != 0 : "chunk size cannot be 0.";

        // Labels to return
        int[] PSi = new int[chunk.size()];

        int[] cluster_ids = getClusterIDs(chunk);

        assert cluster_ids.length == chunk.size() : "KMeans didn't return an ID for each instance.";

        // Calculate cluster_sizes
        Arrays.stream(cluster_ids).forEach(id -> cluster_sizes[id]++);

        int max_pos_labels = (int) (chunk.size() * this.alpha);

        assert sorted_clusters.size() == 0 : "sorted clusters should be empty before each call.";

        // Fill map with <Purity, Cluster>
        for (int i = 0; i < k; i++) {
            final float purity = (float) max_pos_labels / cluster_sizes[i];
            sorted_clusters.add(new SimpleEntry<>(purity, i));
        }

        // Sort clusters on purity (number of genuine positive samples in each cluster)
        Collections.sort(sorted_clusters, (o1, o2) -> Float.compare(o2.getKey(), o1.getKey()));

        int[] labels_per_cluster = new int[k];
        int pos_labels = 0;

        // Calculate the number of labels per cluster to label/set as positive
        for (Entry<Float, Integer> e : sorted_clusters) {
            int size = cluster_sizes[e.getValue()];
            int x = pos_labels + size <= max_pos_labels ? size : max_pos_labels - pos_labels;
            labels_per_cluster[e.getValue()] = x;
            pos_labels += x;
        }

        assert pos_labels == max_pos_labels : "Counted positive labels incorrect.";

        // Fill in labeled array based on number of labels allowed per cluster
        for (int i = 0; i < chunk.size(); i++) {
            if (labels_per_cluster[cluster_ids[i]]-- > 0) {
                PSi[i] = 1; // Mark as positive
            }
        }

        // Reset for next chunk
        Arrays.fill(cluster_sizes, 0);
        sorted_clusters.clear();
        return PSi;
    }
}
