import VLCS.VOCL.*;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.meta.OneClassClassifier;
import weka.core.converters.ArffLoader.ArffReader;

public class VOCLRunner {
    public static void main(String[] args) throws Exception {
        BufferedReader br = null;
        int class_index = 0, attribute_index = 0;
        try {
            br = new BufferedReader(new FileReader(args[0]));
            class_index = Integer.parseInt(args[1]);
            attribute_index = Integer.parseInt(args[2]);
        } catch (Exception e) {
            System.out.println("Usage: <file.arff> <class_index> <attribute_index_from_right_to_left>\n" +
                    "                  i.e class_column_index = 0.");
            System.exit(1);
        }
        ArffReader arff = new ArffReader(br);
        final int chunk_size = 1000;
        VOCL vocl = new VOCL(VOCL.VagueLabelMethod.CLUSTER, class_index, attribute_index);
        vocl.labelStream(arff.getData(), chunk_size);
    }
}
