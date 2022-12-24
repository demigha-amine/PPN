#include <stdio.h>
#include <stdlib.h>
#include "Neural_Network.h"



int main() {

    NeuralNetwork* net = create_network(10, 10, 10, 0.1);
    printf("# of Inputs: %d\n", net->Num_Inputs);
	printf("# of Hidden: %d\n", net->Num_Hidden);
	printf("# of Output: %d\n", net->Num_Outputs);


	printf("Hidden Weights: \n");
	affiche_mat(net->hidden_weights);
    printf("Bias: \n");
	affiche_mat(net->bias);
	printf("Output Weights: \n");
	affiche_mat(net->output_weights);
	
}