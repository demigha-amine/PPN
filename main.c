#include "./Neural_Network/Neural_Network_2.h"

#include <time.h>


#define MAX_SIZE 0
#define LEARNING_RATE 0.1
#define choix 1

int main(int argc, char **argv) {

	if (argc != 5) {
        fprintf(stderr, "Error! Expected:	TrainSize HiddenNodes TestOffset TestSize\n");
        return 1;
    }

	int training_size = atoi(argv[1]);
	int HIDDEN_NODES = atoi(argv[2]);
	int test_offset = atoi(argv[3]);
	int test_size = atoi(argv[4]);

	//CREATE NETWORK

    NeuralNetwork* net = create_network(IMAGE_SIZE, HIDDEN_NODES, OUTPUT_SIZE, LEARNING_RATE);


	// LOAD THE TRAINED MODEL

	// Reseau avec 1 Hidden Layer
	// net->hidden_weights = charger_mat("./Neural_Network/1Hidden/hidden_w");
	// net->hidden_bias = charger_mat("./Neural_Network/1Hidden/hidden_b");
	// net->output_weights = charger_mat("./Neural_Network/1Hidden/output_w");
	// net->output_bias = charger_mat("./Neural_Network/1Hidden/output_b");
	

	// Reseau avec 2 Hidden Layers
    net->hidden_weights = charger_mat("./Neural_Network/TwoHiDDEN/2hidden_w");
	net->hidden_weights_2 = charger_mat("./Neural_Network/TwoHiDDEN/2hidden_w_2");
	net->hidden_bias = charger_mat("./Neural_Network/TwoHiDDEN/2hidden_b");
	net->hidden_bias_2 = charger_mat("./Neural_Network/TwoHiDDEN/2hidden_b_2");
	net->output_weights = charger_mat("./Neural_Network/TwoHiDDEN/2output_w");
	net->output_bias = charger_mat("./Neural_Network/TwoHiDDEN/2output_b");

	
	
	FILE* imageFile = fopen("./mnist_reader/mnist/train-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("./mnist_reader/mnist/train-labels-idx1-ubyte", "r");

	// Read size images from the MAX_SIZE images
	uint8_t* images = readMnistImages(imageFile, MAX_SIZE, training_size);
	uint8_t* labels = readMnistLabels(labelFile, MAX_SIZE, training_size);


	fclose(imageFile);
	fclose(labelFile);


	// 1 TRAINING

	// clock_t trainin_begin = clock();
	// train_batch_imgs(net,images,labels,size,choix);
	// clock_t trainin_end = clock();
    

	// TRAINING TIME
  	// double training_delta = (double) (trainin_end - trainin_begin) / CLOCKS_PER_SEC;
  	

	// 2 TESTING

	imageFile = fopen("./mnist_reader/mnist/t10k-images-idx3-ubyte", "r");
	labelFile = fopen("./mnist_reader/mnist/t10k-labels-idx1-ubyte", "r");

	// Read size images from the MAX_SIZE images
	images = readMnistImages(imageFile, test_offset, test_size);
	labels = readMnistLabels(labelFile, test_offset, test_size);


	fclose(imageFile);
	fclose(labelFile);


    
	double NET_RATE = predict_rate_network(net, images, labels, test_size, choix);

	
	// TRAINING DATASET & HIDDEN NODES PERFORMANCE

	printf("%d; %d; %d; %1.6f\n",
	 training_size,
	 HIDDEN_NODES,
	 test_offset,
	 NET_RATE);


	free_network(net);
    
	
}