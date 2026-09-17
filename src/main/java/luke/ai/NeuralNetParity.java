package luke.ai;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
 /*
  * This neural network calculates
  * the parity bit for a byte
  * 
  * */

public class NeuralNetParity {
		private static final int MAX_EPOCHS = 10000;  // training iterations

		public static void main(String[] args) {

			var input = Nd4j.create(new double [][] 
					{
				{0,1,0,0,0,0,0,0},
				{0,1,0,0,1,0,0,0},
				{1,0,0,0,0,0,0,0},
				{0,0,1,1,0,0,0,0},
				{0,0,1,0,1,1,0,0},
				{1,1,0,1,0,1,0,0},
				{1,0,0,0,0,1,0,1},
				{1,1,1,1,0,1,1,0},
				{1,1,1,1,1,0,1,1},
				{0,0,0,1,0,0,0,1},
				{0,0,0,0,0,0,1,0},
				{0,0,1,1,0,0,0,0},
				{0,0,1,1,1,0,0,0},
				{1,1,1,1,1,1,1,1},
				{1,1,1,0,1,1,1,1},
				{0,0,0,0,0,0,0,0},
				{1,0,0,0,1,0,0,1},
				{1,0,1,0,1,0,0,1}
					});

			var output = Nd4j.create(new double [][] 
					{
				{1},
				{0},
				{1},
				{0},
				{1},
				{0},
				{1},
				{0},
				{1},
				{0},
				{1},
				{0},
				{1},
				{0},
				{1},
				{0},
				{1},
				{0}
					});
			
			var dataset = new DataSet(input, output);
			
			
			var layer0 = new DenseLayer.Builder()
					.nIn(8)
					.nOut(8)
					.activation(Activation.RELU)
					.build();
			
			var layer1 = new DenseLayer.Builder()
					.nIn(8)
					.nOut(4)
					.activation(Activation.RELU)
					.build();
			
			/* RELU activation function:
				if x <= 0 then y = 0 
				else y = x
			*/
			
			var layer2 = new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
					.nIn(4)
					.nOut(1)
					.activation(Activation.SIGMOID)
					.build();
			
			/* SIGMOID activation function:
			 y = 1/ (1 + e^(-x))
			 min = 0
			 max = 1		
		*/
			
			var conf = new NeuralNetConfiguration.Builder()
					.updater(new Sgd(0.1))
					.list()
					.layer(0, layer0)
					.layer(1, layer1)
					.layer(2, layer2)
					.build();
			
			var model = new MultiLayerNetwork(conf);
			model.init();
			
			// Training
			for(int epoch=0; epoch<MAX_EPOCHS; epoch++) {
				model.fit(dataset);
			}
			
			// Inference (with training data)
			var predicted = model.output(input);
			System.out.println("Training Data:");
			System.out.println(predicted);
			
			// Inference (with brand new data)
			var brandNewInput = Nd4j.create(new double [][] 
					{
				{0,1,0,1,0,1,0,1},
				{1,1,0,1,0,1,0,1},
				{1,0,0,1,0,1,0,1},
				{1,0,0,0,1,1,1,1},
				{0,0,0,0,1,1,1,1}
					});
			predicted = model.output(brandNewInput);
			System.out.println("Brand New Data:");
			System.out.println(predicted);

		}

}

