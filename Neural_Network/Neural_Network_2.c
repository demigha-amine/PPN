#include <stdio.h>
#include <stdlib.h>
#include "Neural_Network_2.h"
#include <omp.h>

#define MAX_SIZE 8000

NeuralNetwork* create_network(int input, int hidden, int output, float lr) {

	NeuralNetwork* network = malloc(sizeof(NeuralNetwork));

	network->Num_Inputs = input;
	network->Num_Hidden = hidden;
	network->Num_Hidden_2 = hidden/6;	//
	network->Num_Outputs = output;
	network->LR = lr;
	
	network->hidden_weights = create_mat(hidden, input);
	network->hidden_weights_2 = create_mat(hidden/6, hidden);	//
	network->output_weights = create_mat(output, hidden/6);


	network->hidden_bias = create_mat(hidden,1);
	network->hidden_bias_2 = create_mat(hidden/6,1);	//
	network->output_bias = create_mat(output,1);



	Rand_Matrice(network->hidden_weights);
	Rand_Matrice(network->hidden_weights_2);	//;
	Rand_Matrice(network->output_weights);

	Rand_Bias(network->hidden_bias);
	Rand_Bias(network->hidden_bias_2);		//
	Rand_Bias(network->output_bias);

	return network;
}



void train_network(NeuralNetwork* net, Matrice* input_data, Matrice* output_data,int i) {

	// 1ere Etape (Hidden Layer 1)
	
	Matrice* hidden_inputs	= dotprod(net->hidden_weights, input_data);	
	
	Matrice* z_hidden = add(hidden_inputs, net->hidden_bias);
    
	Matrice* hidden_outputs;
	Matrice* hidden_relu_vec;
    if (i==0)
	{
		//Sigmoid
	    hidden_outputs = apply(sigmoid, z_hidden);
	}
	else {
     	// RELU
	    hidden_relu_vec = apply(relu, z_hidden);
	    hidden_outputs = scale((1.0/mat_max(z_hidden)), hidden_relu_vec);
	}
	

	// 2eme Etape (Hidden Layer 2)

	Matrice* hidden_inputs_2 = dotprod(net->hidden_weights_2, hidden_outputs);

	Matrice* z_hidden_2 = add(hidden_inputs_2, net->hidden_bias_2);

    Matrice* hidden_outputs_2;
	Matrice* hidden_relu_vec_2;
	
	if (i==0)
	{
		//Sigmoid
	    hidden_outputs_2 = apply(sigmoid, z_hidden_2);
	}
	else {
   		// RELU
		hidden_relu_vec_2 = apply(relu, z_hidden_2);
		hidden_outputs_2 = scale((1.0/mat_max(z_hidden_2)), hidden_relu_vec_2);
	}



	// 3eme Etape (Output Layer)
	Matrice* final_inputs = dotprod(net->output_weights, hidden_outputs_2);

	Matrice* z_output = add(final_inputs, net->output_bias);

	//SoftMAX
	Matrice* final_outputs = softmax(z_output);

	


	// // Matrices des erreurs
	Matrice* output_errors = sub(output_data, final_outputs);
	Matrice* hidden_errors = dotprod(transpose(net->output_weights), output_errors);
	Matrice* hidden_errors_2 = dotprod(transpose(net->hidden_weights_2), hidden_errors);
	


	// Backpropagation Processus 

	// Adjusting weights for the output layer	
	// weights : dC/dw = (dz/dw)*(da/dz)/(dC/da)

	//dSOFTMAX
	Matrice* softmax_primed_mat = dSoftmax(final_outputs);


	Matrice* multiplied_mat = mult(output_errors, softmax_primed_mat);

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

	free_mat(softmax_primed_mat);
	free_mat(multiplied_mat);
	free_mat(transposed_mat);
	free_mat(dot_mat);
	free_mat(scaled_mat);
	free_mat(scaled_bias_mat);
	
	


	// // Adjusting weights for the hidden layer 2	
	// // weights : dC/dw = (dz/dw)*(da/dz)/(dC/da)

    Matrice* relu_primed_mat;
	Matrice* sigmoid_primed_mat;
	if (i==0)
	{
		//dSigmoid
	    sigmoid_primed_mat = dSigmoid(hidden_outputs_2);
		multiplied_mat = mult(hidden_errors, sigmoid_primed_mat);
	}
	else {	
		//dRelu
		relu_primed_mat = dRelu(hidden_outputs_2);
		multiplied_mat = mult(hidden_errors, relu_primed_mat);
	}
	
	


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

	if (i==0) free_mat(sigmoid_primed_mat);
	else	free_mat(relu_primed_mat);
	
	free_mat(multiplied_mat);
	free_mat(transposed_mat);
	free_mat(dot_mat);
	free_mat(scaled_mat);
	free_mat(scaled_bias_mat);


	// // Adjusting weights for the hidden layer 1
	// // weights : dC/dw = (dz/dw)*(da/dz)/(dC/da)

	if (i==0)
	{
		//dSigmoid
	    sigmoid_primed_mat = dSigmoid(hidden_outputs);
		multiplied_mat = mult(hidden_errors_2, sigmoid_primed_mat);
	}
	else {	
		//dRelu
		relu_primed_mat = dRelu(hidden_outputs);
		multiplied_mat = mult(hidden_errors_2, relu_primed_mat);
	}
	
	
	
	
	
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
    
	if(i==0) free_mat(sigmoid_primed_mat);
	else free_mat(relu_primed_mat); 


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

    if(i==1){
		free_mat(hidden_relu_vec);
		free_mat(hidden_relu_vec_2);
	}

	free_mat(output_errors);
	free_mat(hidden_errors);
}





// void train_batch_imgs(NeuralNetwork* net, uint8_t* images, uint8_t* labels, int size,int choix)
// {
// 	//#pragma omp parallel for
// 	for (int i =0; i < size; i++)
// 	{
// 		Matrice* IMG = create_mat(IMAGE_SIZE,1);  //IMAGE_SIZE = 784 (CONST)
// 		int k=0;

//         #pragma omp parallel for
// 		for (int j = i * IMAGE_SIZE; j < (i+1) * IMAGE_SIZE; j++)
// 		{   
// 			IMG->data[k] = (float)images[j]/255;
// 			k++;
// 		}
	    
// 	    int index = *(labels + i);  //recupere l'indice de label active

// 		Matrice* output = create_mat(OUTPUT_SIZE, 1);  //OUTPUT_SIZE = 1 (CONST)
// 		output->data[index] = 1.0; // Setting the result
// 		train_network(net, IMG, output,choix);

// 		free_mat(IMG);
// 		free_mat(output);
// 	}

// 	free(images);
// 	free(labels);

// }

void train_batch_imgs(NeuralNetwork* net, uint8_t* images, uint8_t* labels, int size, int choix)
{
    #pragma omp parallel for shared(net, images, labels, size, choix)
    for (int i = 0; i < size; i++)
    {
        Matrice* IMG = create_mat(IMAGE_SIZE, 1);
        int k = 0;

        #pragma omp parallel for shared(IMG, images, k, i)
        for (int j = i * IMAGE_SIZE; j < (i + 1) * IMAGE_SIZE; j++)
        {
            IMG->data[k] = (float)images[j] / 255;
            k++;
        }

        int index = *(labels + i);
        Matrice* output = create_mat(OUTPUT_SIZE, 1);
        output->data[index] = 1.0;

        #pragma omp critical
        {
            train_network(net, IMG, output, choix);
        }

        free_mat(IMG);
        free_mat(output);
    }

    free(images);
    free(labels);
}




void train_batch_imgs_epochs(NeuralNetwork* net, uint8_t* images, uint8_t* labels, int size,int choix)
{
	int epoch = 100;
	int epochs = size / epoch;
	float NET_RATE;

	FILE* imageFile = fopen("./mnist_reader/mnist/t10k-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("./mnist_reader/mnist/t10k-labels-idx1-ubyte", "r");

	// Read size images from the MAX_SIZE images
	uint8_t* test_images = readMnistImages(imageFile, MAX_SIZE, 1000);
	uint8_t* test_labels = readMnistLabels(labelFile, MAX_SIZE, 1000);

	
	fclose(imageFile);
	fclose(labelFile);

	for (int j = 1; j <= epochs; j++)
	{
		NET_RATE = predict_rate_network(net, test_images, test_labels, 1000,choix);
		printf("%1.6f\n", NET_RATE);
		printf("%d; ",j);
		for (int i =(j-1)*epoch; i < j*epoch; i++)
		{
		Matrice* IMG = create_mat(IMAGE_SIZE,1);  //IMAGE_SIZE = 784 (CONST)
        
		////#pragma omp parallel for schedule(dynamic)
		for (int k=0,j = i * IMAGE_SIZE; j < (i+1) * IMAGE_SIZE; j++,k++)
		{   
			IMG->data[k] = (float)images[j]/255;

		}	    
	    int index = *(labels + i);  //recupere l'indice de label active

		Matrice* output = create_mat(OUTPUT_SIZE, 1);  //OUTPUT_SIZE = 1 (CONST)
		output->data[index] = 1.0; // Setting the result

		train_network(net, IMG, output,choix);

		free_mat(IMG);
		free_mat(output);
		}
		
	}
	free(images);
	free(labels);

}



Matrice* predict_network(NeuralNetwork* net, Matrice* IMG, int i) {
	Matrice* hidden_inputs	= dotprod(net->hidden_weights, IMG);

	Matrice* z_hidden = add(hidden_inputs, net->hidden_bias);
 
    Matrice* hidden_outputs;
	
	
    if (i==0)
	{
		//Sigmoid
	    hidden_outputs = apply(sigmoid, z_hidden);
	}
	else {	
		//Relu
		Matrice* hidden_relu_vec = apply(relu, z_hidden);
		hidden_outputs = scale((1.0/mat_max(z_hidden)), hidden_relu_vec);
	}
	

	

	Matrice* hidden_inputs_2 = dotprod(net->hidden_weights_2, hidden_outputs);

	Matrice* z_hidden_2 = add(hidden_inputs_2, net->hidden_bias_2);

    Matrice* hidden_outputs_2;
	
	if (i==0)
	{
		//Sigmoid
	    hidden_outputs_2 = apply(sigmoid, z_hidden_2);
	}
	else {	
		//Relu
	    Matrice* hidden_relu_vec_2 = apply(relu, z_hidden_2);
	    hidden_outputs_2 = scale((1.0/mat_max(z_hidden_2)), hidden_relu_vec_2);
	}
	

	Matrice* final_inputs = dotprod(net->output_weights, hidden_outputs_2);

	Matrice* z_output = add(final_inputs, net->output_bias);

	//Softmax
	Matrice* final_outputs = softmax(z_output);

	return final_outputs;
}


// float predict_rate_network(NeuralNetwork* net, uint8_t* images, uint8_t* labels, int size,int choix) {
// 	
//      int img_correct = 0;
  
//      //#pragma omp parallel for reduction(+:img_correct)
// 	for (int i = 0; i < size; i++) {
// 		Matrice* IMG = create_mat(IMAGE_SIZE,1);  //IMAGE_SIZE = 784 (CONST)

// 		for (int k=0,j = i * IMAGE_SIZE; j < (i+1) * IMAGE_SIZE; j++,k++)
// 		{   
// 			IMG->data[k] = (float)images[j]/255;

// 		}			// recuperer la matrice qui contient les activations de l'image de test

// 	    int index = *(labels + i);		// recuperer le vrai label de l'image de test

// 		Matrice* prediction = predict_network(net, IMG,choix); // la couche de sortie evaluée par notre reseau pour une image de test
		
// 		if (mat_argmax(prediction) == index) {
		
// 			// printf("Label de sortie = %d\n",index);
// 			img_correct++;  //compteur pour les tests validés 
// 		}
// 		free_mat(IMG);
// 		free_mat(prediction);
// 	}
		
// 	// printf("***********************************\n");
// 	// printf("Nombre des predictions correctes est: %d\n",img_correct);
// 	// printf("Nombre des predictions incorrectes est: %d\n",(size-img_correct));

// 	return (1.0 * img_correct / size) * 100; //calculer le pourcentage des tests validés
// }



float predict_rate_network(NeuralNetwork* net, uint8_t* images, uint8_t* labels, int size,int choix) {
	int img_correct = 0;

	#pragma omp parallel for shared(net, images, labels, size, choix)

	for (int i = 0; i < size; i++) 
	{
		Matrice* IMG = create_mat(IMAGE_SIZE,1);  //IMAGE_SIZE = 784 (CONST)
		int k=0;
        
		#pragma omp parallel for shared(IMG, images, k, i)

		for (int j = i * IMAGE_SIZE; j < (i+1) * IMAGE_SIZE; j++)
		{   
			IMG->data[k] = (float)images[j]/255;
			k++;

		}

	    int index = *(labels + i);
		Matrice* prediction;

        #pragma omp critical

        {
			prediction = predict_network(net, IMG,choix);
		}

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