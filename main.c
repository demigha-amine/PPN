#include "./Neural_Network/Neural_Network.h"


#define size_test 3
#define OFFSET 8000



int main(void) {

    // creattion de notre reseau de neuron
    NeuralNetwork* net = create_network(IMAGE_SIZE, HIDDEN_NODES, OUTPUT_SIZE, LEARNING_RATE);
	affiche_network(net);


    printf("\n********* Les Tests *******************\n");
    

    //charger les resultats d'entrainement
    net->hidden_weights = charger_mat("./Neural_Network/hidden_w");
	net->hidden_bias = charger_mat("./Neural_Network/hidden_b");
	net->output_weights = charger_mat("./Neural_Network/output_w");
	net->output_bias = charger_mat("./Neural_Network/output_b");



    // LES IMAGES DE TESTS + LABELS
	FILE* imageFile = fopen("./mnist_reader/mnist/t10k-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("./mnist_reader/mnist/t10k-labels-idx1-ubyte", "r");

	// Read size images from the OFFSET images
	uint8_t* images = readMnistImages(imageFile, OFFSET, size_test);
	uint8_t* labels = readMnistLabels(labelFile, OFFSET, size_test);


	fclose(imageFile);
	fclose(labelFile);

    //Le pourcentage des tests
	double NET_RATE = predict_rate_network(net, images, labels, size_test);
	
	//afficher le resultat en %
	printf("\n***********************************\n");
	printf("Network predict = %1.6f %%\n",NET_RATE);
	printf("***********************************\n");

	
	free_network(net);
    
}