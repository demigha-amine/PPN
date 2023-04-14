#include <stdio.h>
#include <stdlib.h>
#include "Neural_Network.h"



NeuralNetwork* create_network(int input, int hidden, int output, double lr) {

	NeuralNetwork* network = malloc(sizeof(NeuralNetwork));

	network->Num_Inputs = input;
	network->Num_Hidden = hidden;
	network->Num_Hidden_2 = hidden/2;	//
	network->Num_Outputs = output;
	network->LR = lr;
	
	network->hidden_weights = create_mat(hidden, input);
	network->hidden_weights_2 = create_mat(hidden/2, hidden);	//
	network->output_weights = create_mat(output, hidden/2);


	network->hidden_bias = create_mat(hidden,1);
	network->hidden_bias_2 = create_mat(hidden/2,1);	//
	network->output_bias = create_mat(output,1);



	Rand_Matrice(network->hidden_weights);
	Rand_Matrice(network->hidden_weights_2);	//;
	Rand_Matrice(network->output_weights);

	Rand_Bias(network->hidden_bias);
	Rand_Bias(network->hidden_bias_2);		//
	Rand_Bias(network->output_bias);

	return network;
}



void train_network(NeuralNetwork* net, Matrice* input_data, Matrice* output_data) {

	// 1ere Etape (Hidden Layer 1)
	
	Matrice* hidden_inputs	= dotprod(net->hidden_weights, input_data);
	

	Matrice* z_hidden = add(hidden_inputs, net->hidden_bias);

	Matrice* hidden_outputs = apply(sigmoid, z_hidden);


	// 2eme Etape (Hidden Layer 2)

	Matrice* hidden_inputs_2	= dotprod(net->hidden_weights_2, hidden_outputs);

	Matrice* z_hidden_2 = add(hidden_inputs_2, net->hidden_bias_2);

	Matrice* hidden_outputs_2 = apply(sigmoid, z_hidden_2);

	
	

	// 3eme Etape (Output Layer)
	Matrice* final_inputs = dotprod(net->output_weights, hidden_outputs_2);

	Matrice* z_output = add(final_inputs, net->output_bias);

	Matrice* final_outputs = apply(sigmoid, z_output);

	


	// // Matrices des erreurs
	Matrice* output_errors = sub(output_data, final_outputs);
	Matrice* hidden_errors = dotprod(transpose(net->output_weights), output_errors);
	Matrice* hidden_errors_2 = dotprod(transpose(net->hidden_weights_2), hidden_errors);
	


	// Backpropagation Processus 

	// Adjusting weights for the output layer	
	// weights : dC/dw = (dz/dw)*(da/dz)/(dC/da)



	Matrice* sigmoid_primed_mat = dSigmoid(final_outputs);
	Matrice* multiplied_mat = mult(output_errors, sigmoid_primed_mat);

	//
	Matrice* scaled_bias_mat = scale(net->LR, multiplied_mat);
	Matrice* added_bias_mat = add(net->output_bias, scaled_bias_mat);


	//
	Matrice* transposed_mat = transpose(hidden_outputs_2);
	Matrice* dot_mat = dotprod(multiplied_mat, transposed_mat);
	Matrice* scaled_mat = scale(net->LR, dot_mat);
	Matrice* added_mat = add(net->output_weights, scaled_mat);




	free_mat(net->output_bias);			 
	net->output_bias = added_bias_mat;		// Remplacement par la matrice correcte


	free_mat(net->output_weights);			 // free the old output weights matrix before replacing
	net->output_weights = added_mat;		// Remplacement par la matrice correcte

	free_mat(sigmoid_primed_mat);
	free_mat(multiplied_mat);
	free_mat(transposed_mat);
	free_mat(dot_mat);
	free_mat(scaled_mat);
	free_mat(scaled_bias_mat);
	
	


	// // Adjusting weights for the hidden layer 2	
	// // weights : dC/dw = (dz/dw)*(da/dz)/(dC/da)

	
	
	sigmoid_primed_mat = dSigmoid(hidden_outputs_2);
	multiplied_mat = mult(hidden_errors, sigmoid_primed_mat);


	//
	scaled_bias_mat = scale(net->LR, multiplied_mat);
	added_bias_mat = add(net->hidden_bias_2, scaled_bias_mat);



	//
	transposed_mat = transpose(hidden_outputs);
	dot_mat = dotprod(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->LR, dot_mat);
	added_mat = add(net->hidden_weights_2, scaled_mat);

			


	free_mat(net->hidden_bias_2); 				// free the old hidden weights matrix before replacing
	net->hidden_bias_2 = added_bias_mat; 			// Remplacement par la matrice correcte


	free_mat(net->hidden_weights_2); 				// free the old hidden weights matrix before replacing
	net->hidden_weights_2 = added_mat; 			// Remplacement par la matrice correcte

	free_mat(sigmoid_primed_mat);
	free_mat(multiplied_mat);
	free_mat(transposed_mat);
	free_mat(dot_mat);
	free_mat(scaled_mat);
	free_mat(scaled_bias_mat);


	// // Adjusting weights for the hidden layer 1
	// // weights : dC/dw = (dz/dw)*(da/dz)/(dC/da)

	
	
	sigmoid_primed_mat = dSigmoid(hidden_outputs);
	
	multiplied_mat = mult(hidden_errors_2, sigmoid_primed_mat);
	
	//
	scaled_bias_mat = scale(net->LR, multiplied_mat);
	added_bias_mat = add(net->hidden_bias, scaled_bias_mat);



	//
	transposed_mat = transpose(input_data);
	dot_mat = dotprod(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->LR, dot_mat);
	added_mat = add(net->hidden_weights, scaled_mat);

	

	free_mat(net->hidden_bias); 				// free the old hidden weights matrix before replacing
	net->hidden_bias = added_bias_mat; 			// Remplacement par la matrice correcte


	free_mat(net->hidden_weights); 				// free the old hidden weights matrix before replacing
	net->hidden_weights = added_mat; 			// Remplacement par la matrice correcte

	free_mat(sigmoid_primed_mat);
	free_mat(multiplied_mat);
	free_mat(transposed_mat);
	free_mat(dot_mat);
	free_mat(scaled_mat);
	free_mat(scaled_bias_mat);


	// Free matrices
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

		Matrice* output = create_mat(OUTPUT_SIZE, 1);  //OUTPUT_SIZE = 1 (CONST)
		output->data[index] = 1.0; // Setting the result

		train_network(net, IMG, output);

		free_mat(IMG);
		free_mat(output);
	}

	free(images);
	free(labels);

}



Matrice* network_predict(NeuralNetwork* net, Matrice* IMG) {
	Matrice* hidden_inputs	= dotprod(net->hidden_weights, IMG);

	Matrice* z_hidden = add(hidden_inputs, net->hidden_bias);
	Matrice* hidden_outputs = apply(sigmoid, z_hidden);

	Matrice* hidden_inputs_2	= dotprod(net->hidden_weights_2, hidden_outputs);

	Matrice* z_hidden_2 = add(hidden_inputs_2, net->hidden_bias_2);
	Matrice* hidden_outputs_2 = apply(sigmoid, z_hidden_2);

	Matrice* final_inputs = dotprod(net->output_weights, hidden_outputs_2);

	Matrice* z_output = add(final_inputs, net->output_bias);
	Matrice* final_outputs = apply(sigmoid, z_output);

	return final_outputs;
}



double network_predict_rate(NeuralNetwork* net, uint8_t* images, uint8_t* labels, int size) {
	int img_correct = 0;
	for (int i = 0; i < size; i++) {
		Matrice* IMG = create_mat(IMAGE_SIZE,1);  //IMAGE_SIZE = 784 (CONST)
 
		for (int k=0,j = i * IMAGE_SIZE; j < (i+1) * IMAGE_SIZE; j++,k++)
		{   
			IMG->data[k] = (double)images[j]/255;

		}
	    int index = *(labels + i);

		Matrice* prediction = network_predict(net, IMG);
		// affiche_mat(prediction);
		if (mat_argmax(prediction) == index) {
			img_correct++;
		}
		free_mat(IMG);
		free_mat(prediction);
	}
	return (1.0 * img_correct / size) * 100;
}



void affiche_network(NeuralNetwork* net) {
	printf("Number of Inputs: %d\n", net->Num_Inputs);
	printf("Number of Hidden Nodes: %d\n", net->Num_Hidden);
	printf("Number of Outputs: %d\n", net->Num_Outputs);
	printf("Learning Rate = %f\n", net->LR);
}


void free_network(NeuralNetwork *net) {
	free_mat(net->hidden_weights);
	free_mat(net->output_weights);
	free(net);
	net = NULL;
}