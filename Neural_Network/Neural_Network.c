#include "Neural_Network.h"



NeuralNetwork* create_network(int input, int hidden, int output, double lr) {

	NeuralNetwork* network = malloc(sizeof(NeuralNetwork));

	network->Num_Hidden = hidden;
	network->Num_Outputs = output;
	network->LR = lr;

	network->hidden_weights = create_mat(hidden, input);
	network->output_weights = create_mat(output, hidden);
	network->bias = create_mat(hidden,1);

	Rand_Matrice(network->hidden_weights);
	Rand_Matrice(network->output_weights);
	Rand_Matrice(network->bias);

	return network;
}



void train_network(NeuralNetwork* net, Matrice* input_data, Matrice* output_data) {

	// 1ere Etape
	Matrice* hidden_inputs	= dotprod(net->hidden_weights, input_data);
	Matrice* z = add(hidden_inputs, net->bias);
	Matrice* hidden_outputs = apply(sigmoid, z);

	// 2eme Etape
	Matrice* final_inputs = dotprod(net->output_weights, hidden_outputs);
	Matrice* final_outputs = apply(sigmoid, final_inputs);

	// Matrices des erreurs
	Matrice* output_errors = sub(output_data, final_outputs);
	Matrice* hidden_errors = dotprod(transpose(net->output_weights), output_errors);

	

	// Free matrices
	free_mat(hidden_inputs);
	free_mat(z);
	free_mat(hidden_outputs);
	free_mat(final_inputs);
	free_mat(final_outputs);
	free_mat(output_errors);
	free_mat(hidden_errors);
}