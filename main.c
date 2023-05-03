#include "./Neural_Network/Neural_Network_2.h"

#include <time.h>


#define MAX_SIZE 0
#define LEARNING_RATE 0.1
#define choix 1

int main(int argc, char **argv) {

	if (argc != 4) {
        fprintf(stderr, "Error! Expecting :    ./exe	Training size	Hidden Nodes	Test Offset\n");
        return 1;
    }

	int size = atoi(argv[1]);
	int HIDDEN_NODES = atoi(argv[2]);
	int test_offset = atoi(argv[3]);

	//CREATE NETWORK

    NeuralNetwork* net = create_network(IMAGE_SIZE, HIDDEN_NODES, OUTPUT_SIZE, LEARNING_RATE);
	
    net->hidden_weights = charger_mat("./Neural_Network/hidden_w");
	net->hidden_weights_2 = charger_mat("./Neural_Network/hidden2_w");
	net->hidden_bias = charger_mat("./Neural_Network/hidden_b");
	net->hidden_bias_2 = charger_mat("./Neural_Network/hidden2_b");
	net->output_weights = charger_mat("./Neural_Network/output_w");
	net->output_bias = charger_mat("./Neural_Network/output_b");

	FILE* imageFile = fopen("./mnist_reader/mnist/train-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("./mnist_reader/mnist/train-labels-idx1-ubyte", "r");

	// Read size images from the MAX_SIZE images
	uint8_t* images = readMnistImages(imageFile, MAX_SIZE, size);
	uint8_t* labels = readMnistLabels(labelFile, MAX_SIZE, size);


	fclose(imageFile);
	fclose(labelFile);


	// 1 TRAINING

	clock_t trainin_begin = clock();
	train_batch_imgs(net,images,labels,size,choix);
	clock_t trainin_end = clock();
    

	// TRAINING TIME
  	double training_delta = (double) (trainin_end - trainin_begin) / CLOCKS_PER_SEC;
  	

	// 2 TESTING

	imageFile = fopen("./mnist_reader/mnist/t10k-images-idx3-ubyte", "r");
	labelFile = fopen("./mnist_reader/mnist/t10k-labels-idx1-ubyte", "r");

	// Read size images from the MAX_SIZE images
	images = readMnistImages(imageFile, test_offset, 3000);
	labels = readMnistLabels(labelFile, test_offset, 3000);


	fclose(imageFile);
	fclose(labelFile);


    
	double NET_RATE = predict_rate_network(net, images, labels, 3000,choix);

	
	// TRAINING DATASET & HIDDEN NODES PERFORMANCE

	printf("%d; %d; %f; %1.6f\n",
	 size,
	 HIDDEN_NODES,
	 training_delta,
	 NET_RATE);


	free_network(net);
    
	
}