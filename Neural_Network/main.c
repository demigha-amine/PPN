#include "Neural_Network.h"
#include <time.h>


#define size_train 8000
#define OFFSET 8000



int main(void) {

    NeuralNetwork* net = create_network(IMAGE_SIZE, HIDDEN_NODES, OUTPUT_SIZE, LEARNING_RATE); 
	affiche_network(net);

    // LES IMAGES D'ENTRAINEMENT
	FILE* imageFile = fopen("../mnist_reader/mnist/train-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("../mnist_reader/mnist/train-labels-idx1-ubyte", "r");

	// Read size images from the OFFSET
	uint8_t* images = readMnistImages(imageFile, OFFSET, size_train);
	uint8_t* labels = readMnistLabels(labelFile, OFFSET, size_train);


	fclose(imageFile);
	fclose(labelFile);
	
	printf("\n*********Debut d'entrainement*******************\n");

	//calculer le temps d'entrainements
    clock_t t = clock();
	train_batch_imgs(net,images,labels,size_train);
	clock_t r = clock();

	printf("Temps d'entrainement = %f\n", (double)(r-t)/CLOCKS_PER_SEC);

	printf("\n*********Fin d'entrainement*******************\n");
    
	//sauvgarder les resultats de tests dans des fichiers
	save_mat(net->hidden_weights,"hidden_w");
	save_mat(net->output_weights,"output_w");
	save_mat(net->hidden_bias,"hidden_b");
	save_mat(net->output_bias,"output_b");
	
	free_network(net);
    
}