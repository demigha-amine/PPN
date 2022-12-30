#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "../mnist_reader/mnist_reader.h"
#include "Neural_Network.h"

#define size 1000
#define MAX_SIZE 10000


int main(void) {

    NeuralNetwork* net = create_network(IMAGE_SIZE, HIDDEN_NODES, OUTPUT_SIZE, LEARNING_RATE);
    printf("Number of Inputs: %d\n", net->Num_Inputs);
	printf("Number of Hidden Nodes: %d\n", net->Num_Hidden);
	printf("Number of Outputs: %d\n", net->Num_Outputs);


	


	printf("**********After training:*********** \n\n\n\n\n\n\n");



	FILE* imageFile = fopen("../mnist_reader/mnist/train-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("../mnist_reader/mnist/train-labels-idx1-ubyte", "r");

	// Read size = 15 images from the 50 image
	uint8_t* images = readMnistImages(imageFile, MAX_SIZE, size);
	uint8_t* labels = readMnistLabels(labelFile, MAX_SIZE, size);


	fclose(imageFile);
	fclose(labelFile);
    
	train_batch_imgs(net,images,labels,size);
	
    
	
}