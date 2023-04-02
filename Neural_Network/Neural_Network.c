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
	network->hidden_bias = create_mat(hidden,1);
	network->output_bias = create_mat(output,1);

	Rand_Matrice(network->hidden_weights);
	Rand_Matrice(network->output_weights);
	Rand_Bias(network->hidden_bias);
	Rand_Bias(network->output_bias);

	return network;
}



void train_network(NeuralNetwork* net, Matrice* input_data, Matrice* output_data) {

	// 1ere Etape (Hidden Layer)
	
	Matrice* hidden_inputs	= dotprod(net->hidden_weights, input_data);
	Matrice* z_hidden = add(hidden_inputs, net->hidden_bias);
	Matrice* hidden_outputs = apply(sigmoid, z_hidden);
	

	// 2eme Etape (Output Layer)
	Matrice* final_inputs = dotprod(net->output_weights, hidden_outputs);
	Matrice* z_output = add(final_inputs, net->output_bias);
	Matrice* final_outputs = apply(sigmoid, z_output);

	


	// Matrices des erreurs
	Matrice* output_errors = sub(output_data, final_outputs);
	Matrice* hidden_errors = dotprod(transpose(net->output_weights), output_errors);
	



	//*****Backpropagation Processus*****//


	// Adjusting weights and biases for the output layer	
	// weights : dC/dw = (dz/dw)*(da/dz)/(dC/da)
	// biases : dC/db = (dz/db)*(da/dz)/(dC/da)


	Matrice* sigmoid_primed_mat = dSigmoid(final_outputs);
	Matrice* multiplied_mat = mult(output_errors, sigmoid_primed_mat);

	// Adjusting Biases
	Matrice* scaled_bias_mat = scale(net->LR, multiplied_mat);
	Matrice* added_bias_mat = add(net->output_bias, scaled_bias_mat);

	// Adjusting weights
	Matrice* transposed_mat = transpose(hidden_outputs);
	Matrice* dot_mat = dotprod(multiplied_mat, transposed_mat);
	Matrice* scaled_mat = scale(net->LR, dot_mat);
	Matrice* added_mat = add(net->output_weights, scaled_mat);


	free_mat(net->output_bias);			 	// free the old output bias matrix before replacing
	net->output_bias = added_bias_mat;		// Remplacement par la matrice correcte


	free_mat(net->output_weights);			 // free the old output weights matrix before replacing
	net->output_weights = added_mat;		// Remplacement par la matrice correcte

	
	// Free les Matrices pour les utiliser dans la prochaine etape
	free_mat(sigmoid_primed_mat);
	free_mat(multiplied_mat);
	free_mat(transposed_mat);
	free_mat(dot_mat);
	free_mat(scaled_mat);
	free_mat(scaled_bias_mat);
	
	


	// Adjusting weights and biases for the hidden layer	
	// weights : dC/dw = (dz/dw)*(da/dz)/(dC/da)
	// biases : dC/db = (dz/db)*(da/dz)/(dC/da)

	
	
	sigmoid_primed_mat = dSigmoid(hidden_outputs);
	multiplied_mat = mult(hidden_errors, sigmoid_primed_mat);

	// Adjusting Biases
	scaled_bias_mat = scale(net->LR, multiplied_mat);
	added_bias_mat = add(net->hidden_bias, scaled_bias_mat);

	// Adjusting weights
	transposed_mat = transpose(input_data);
	dot_mat = dotprod(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->LR, dot_mat);
	added_mat = add(net->hidden_weights, scaled_mat);


	free_mat(net->hidden_bias); 				// free the old hidden bias matrix before replacing
	net->hidden_bias = added_bias_mat; 			// Remplacement par la matrice correcte


	free_mat(net->hidden_weights); 				// free the old hidden weights matrix before replacing
	net->hidden_weights = added_mat; 			// Remplacement par la matrice correcte


	// Free les matrices
	free_mat(sigmoid_primed_mat);
	free_mat(multiplied_mat);
	free_mat(transposed_mat);
	free_mat(dot_mat);
	free_mat(scaled_mat);
	free_mat(scaled_bias_mat);
	free_mat(hidden_inputs);
	free_mat(hidden_outputs);
	free_mat(final_inputs);
	free_mat(final_outputs);
	free_mat(z_hidden);
	free_mat(z_output);
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

		Matrice* output = create_mat(OUTPUT_SIZE, 1);  //OUTPUT_SIZE = 10 (CONST)
		output->data[index] = 1.0; // Setting the result

		train_network(net, IMG, output);

		free_mat(IMG);
		free_mat(output);
	}

	free(images);
	free(labels);

}



Matrice* predict_network(NeuralNetwork* net, Matrice* IMG) {
	
	//Appliquer propagation sur l'image
	Matrice* hidden_inputs	= dotprod(net->hidden_weights, IMG);
	Matrice* z_hidden = add(hidden_inputs, net->hidden_bias);
	Matrice* hidden_outputs = apply(sigmoid, z_hidden);

	Matrice* final_inputs = dotprod(net->output_weights, hidden_outputs);
	Matrice* z_output = add(final_inputs, net->output_bias);
	Matrice* final_outputs = apply(sigmoid, z_output);
	
	return final_outputs;
}


double predict_rate_network(NeuralNetwork* net, uint8_t* images, uint8_t* labels, int size) {
	int img_correct = 0;
	// printf("\n***********************************\n");

	for (int i = 0; i < size; i++) {
		Matrice* IMG = create_mat(IMAGE_SIZE,1);  //IMAGE_SIZE = 784 (CONST)

		for (int k=0,j = i * IMAGE_SIZE; j < (i+1) * IMAGE_SIZE; j++,k++)
		{   
			IMG->data[k] = (double)images[j]/255;

		}			// recuperer la matrice qui contient les activations de l'image de test

	    int index = *(labels + i);		// recuperer le vrai label de l'image de test

		Matrice* prediction = predict_network(net, IMG); // la couche de sortie evaluée par notre reseau pour une image de test
		
		if (mat_argmax(prediction) == index) {
		
			// printf("Label de sortie = %d\n",index);
			img_correct++;  //compteur pour les tests validés 
		}
		free_mat(IMG);
		free_mat(prediction);
	}
		
	// printf("***********************************\n");
	// printf("Nombre des predictions correctes est: %d\n",img_correct);
	// printf("Nombre des predictions incorrectes est: %d\n",(size-img_correct));

	return (1.0 * img_correct / size) * 100; //calculer le pourcentage des tests validés
}



void affiche_network(NeuralNetwork* net) {
	printf("\n********* Network *******************\n");
	printf("Number of Inputs: %d\n", net->Num_Inputs);
	printf("Number of Hidden Nodes: %d\n", net->Num_Hidden);
	printf("Number of Outputs: %d\n", net->Num_Outputs);
	printf("Learning Rate: %f\n", net->LR);
	printf("***************************************\n");

}


void free_network(NeuralNetwork *net) {
	free_mat(net->hidden_weights);
	free_mat(net->output_weights);
	free_mat(net->hidden_bias);
	free_mat(net->output_bias);
	free(net);
	net = NULL;
}