#include "Neural_Network.h"


#define size_train 8000
#define size_test 1000
#define OFFSET 8000



int main(void) {

    NeuralNetwork* net = create_network(IMAGE_SIZE, HIDDEN_NODES, OUTPUT_SIZE, LEARNING_RATE);
  
	affiche_network(net);

    // LES IMAGES D'ENTRAINEMENT
	FILE* imageFile = fopen("../mnist_reader/mnist/train-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("../mnist_reader/mnist/train-labels-idx1-ubyte", "r");

	// Read size (1000) images from the OFFSET = 0 images
	uint8_t* images = readMnistImages(imageFile, OFFSET, size_train);
	uint8_t* labels = readMnistLabels(labelFile, OFFSET, size_train);


	fclose(imageFile);
	fclose(labelFile);
	

	train_batch_imgs(net,images,labels,size_train);

    printf("*********Fin train*******************\n");



    // LES IMAGES DE TESTS
	FILE* image2File = fopen("../mnist_reader/mnist/t10k-images-idx3-ubyte", "r");
	FILE* label2File = fopen("../mnist_reader/mnist/t10k-labels-idx1-ubyte", "r");

	// Read size images from the OFFSET images
	uint8_t* images2 = readMnistImages(image2File, OFFSET, size_test);
	uint8_t* labels2 = readMnistLabels(label2File, OFFSET, size_test);


	fclose(image2File);
	fclose(label2File);

   
	double NET_RATE = predict_rate_network(net, images2, labels2, size_test);
	
	printf("network predict = %1.6f %%\n",NET_RATE);
	
	free_network(net);
    
	
}