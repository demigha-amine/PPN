#include <stdio.h>
#include <stdlib.h>
#include "Neural_Network.h"



int main() {

    NeuralNetwork* net = create_network(20, 16, 10, 0.1);
    printf("Number of Inputs: %d\n", net->Num_Inputs);
	printf("Number of Hidden Nodes: %d\n", net->Num_Hidden);
	printf("Number of Outputs: %d\n", net->Num_Outputs);


	printf("********Before training**********\n\n\n\n\n");

	printf("Hidden Weights: \n");
	affiche_mat(net->hidden_weights);
    printf("Hidden Bias: \n");
	affiche_mat(net->hidden_bias);
	printf("Output Weights: \n");
	affiche_mat(net->output_weights);
	printf("Output bias: \n");
	affiche_mat(net->output_bias);

	Matrice* input = create_mat(20,1);
	Rand_Matrice(input);
	

	Matrice* output = create_mat(10,1);
	output->data[2] = 1.0;
	

	train_network(net, input, output);

	printf("**********After training:*********** \n\n\n\n\n\n\n");

	printf("Hidden Weights: \n");
	affiche_mat(net->hidden_weights);
    printf("Hidden Bias: \n");
	affiche_mat(net->hidden_bias);
	printf("Output Weights: \n");
	affiche_mat(net->output_weights);
	printf("Output bias: \n");
	affiche_mat(net->output_bias);
	
}