#include <stdio.h>
#include <stdlib.h>
#include "Neural_Network.h"



NeuralNetwork* create_network(int input, int hidden, int output, double lr) {

	NeuralNetwork* network = malloc(sizeof(NeuralNetwork));

	network->Num_Inputs = input;
	network->Num_Hidden = hidden;
	network->Num_Outputs = output;
	network->LR = lr;
	
	network->hidden_weights = create_mat(hidden, input);
	network->output_weights = create_mat(output, hidden);

	Rand_Matrice(network->hidden_weights);
	Rand_Matrice(network->output_weights);

	return network;
}



void train_network(NeuralNetwork* net, Matrice* input_data, Matrice* output_data) {

	// 1ere Etape (Hidden Layer)
	
	Matrice* hidden_inputs	= dotprod(net->hidden_weights, input_data);
	Matrice* hidden_outputs = apply(sigmoid, hidden_inputs);
	

	// 2eme Etape (Output Layer)
	Matrice* final_inputs = dotprod(net->output_weights, hidden_outputs);
	Matrice* final_outputs = apply(sigmoid, final_inputs);

	


	// // Matrices des erreurs
	Matrice* output_errors = sub(output_data, final_outputs);
	Matrice* hidden_errors = dotprod(transpose(net->output_weights), output_errors);
	



	// Backpropagation Processus

	// Adjusting weights for the output layer	
	// weights : dC/dw = (dz/dw)*(da/dz)/(dC/da)



	Matrice* sigmoid_primed_mat = dSigmoid(final_outputs);
	Matrice* multiplied_mat = mult(output_errors, sigmoid_primed_mat);
	Matrice* transposed_mat = transpose(hidden_outputs);
	Matrice* dot_mat = dotprod(multiplied_mat, transposed_mat);
	Matrice* scaled_mat = scale(net->LR, dot_mat);
	Matrice* added_mat = add(net->output_weights, scaled_mat);


	free_mat(net->output_weights);			 // free the old output weights matrix before replacing
	net->output_weights = added_mat;		// Remplacement par la matrice correcte

	free_mat(sigmoid_primed_mat);
	free_mat(multiplied_mat);
	free_mat(transposed_mat);
	free_mat(dot_mat);
	free_mat(scaled_mat);
	
	


	// // Adjusting weights for the hidden layer	
	// // weights : dC/dw = (dz/dw)*(da/dz)/(dC/da)

	
	
	sigmoid_primed_mat = dSigmoid(hidden_outputs);
	multiplied_mat = mult(hidden_errors, sigmoid_primed_mat);
	transposed_mat = transpose(input_data);
	dot_mat = dotprod(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->LR, dot_mat);
	added_mat = add(net->hidden_weights, scaled_mat);

	free_mat(net->hidden_weights); 				// free the old hidden weights matrix before replacing
	net->hidden_weights = added_mat; 			// Remplacement par la matrice correcte

	free_mat(sigmoid_primed_mat);
	free_mat(multiplied_mat);
	free_mat(transposed_mat);
	free_mat(dot_mat);
	free_mat(scaled_mat);

	// Free matrices
	free_mat(hidden_inputs);
	free_mat(hidden_outputs);
	free_mat(final_inputs);
	free_mat(final_outputs);
	free_mat(output_errors);
	free_mat(hidden_errors);
}



void train_batch_imgs(NeuralNetwork* net, uint8_t* images, uint8_t* labels, int size)
{
	
	for (int i =0; i < size; i++)
	{

		Matrice* IMG = create_mat(IMAGE_SIZE,1);  //IMAGE_SIZE = 784 (CONST)
 
		for (int k=0,j = i * IMAGE_SIZE; j < (i+1) * IMAGE_SIZE; j++,k++)
		{   
			IMG->data[k] = (double)images[j]/255;

		}

	    
	    int index = *(labels + i);  //recupere l'indice de label active

		Matrice* output = create_mat(OUTPUT_SIZE, 1);  //OUTPUT_SIZE = 1 (CONST)
		output->data[index] = 1.0; // Setting the result

		train_network(net, IMG, output);

		free_mat(IMG);
		free_mat(output);
	}

	free(images);
	free(labels);

}