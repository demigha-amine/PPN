#include "./Neural_Network/Neural_Network.h"
#include <time.h>


#define MAX_SIZE 8000
#define LEARNING_RATE 0.1


int main(int argc, char **argv) {

	if (argc != 3) {
        fprintf(stderr, "Error! Expecting :    ./exe	Training size	Hidden Nodes\n");
        return 1;
    }

	int size = atoi(argv[1]);
	int HIDDEN_NODES = atoi(argv[2]);

	//CREATE NETWORK

    NeuralNetwork* net = create_network(IMAGE_SIZE, HIDDEN_NODES, OUTPUT_SIZE, LEARNING_RATE);
  


	FILE* imageFile = fopen("./mnist_reader/mnist/train-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("./mnist_reader/mnist/train-labels-idx1-ubyte", "r");

	// Read size images from the MAX_SIZE images
	uint8_t* images = readMnistImages(imageFile, MAX_SIZE, size);
	uint8_t* labels = readMnistLabels(labelFile, MAX_SIZE, size);


	fclose(imageFile);
	fclose(labelFile);


	// 1 TRAINING

	clock_t trainin_begin = clock();
	train_batch_imgs(net,images,labels,size);
	clock_t trainin_end = clock();


	// TRAINING TIME
  	double training_delta = (double) (trainin_end - trainin_begin) / CLOCKS_PER_SEC;
  	

	// 2 TESTING

	imageFile = fopen("./mnist_reader/mnist/t10k-images-idx3-ubyte", "r");
	labelFile = fopen("./mnist_reader/mnist/t10k-labels-idx1-ubyte", "r");

	// Read size images from the MAX_SIZE images
	images = readMnistImages(imageFile, MAX_SIZE, 1000);
	labels = readMnistLabels(labelFile, MAX_SIZE, 1000);


	fclose(imageFile);
	fclose(labelFile);


    
	double NET_RATE = predict_rate_network(net, images, labels, 1000);

	
	// TRAINING DATASET & HIDDEN NODES PERFORMANCE

	printf("%d; %d; %f; %1.6f\n",
	 size,
	 HIDDEN_NODES,
	 training_delta,
	 NET_RATE);


	free_network(net);
    
	
}