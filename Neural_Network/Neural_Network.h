#pragma once
#include "../Activation/activations.h"
#include "../Matrix/oper_mat.h"
#include "../mnist_reader/mnist_reader.h"

#define IMAGE_SIZE 784
#define OUTPUT_SIZE 10
#define HIDDEN_NODES 300
#define LEARNING_RATE 0.1


typedef struct {

	int Num_Inputs;		//Nbr des elements d'entrees
	int Num_Hidden;		//Nbr des noeuds caches
	int Num_Outputs;	//Nbr des elements de sortie
	double LR;			//Learning Rate


	Matrice* hidden_weights;		// Matrice des weights caches
	Matrice* output_weights;		// Matrice des weights de sortie
	Matrice* hidden_bias;			// Matrice des bias caches		
	Matrice* output_bias;			// Matrice des bias de sortie

} NeuralNetwork;


// Fonction de creation d'un reseau de neurones
NeuralNetwork* create_network(int input, int hidden, int output, double lr);

// Fonction pour entrainer le reseau de neurones
void train_network(NeuralNetwork* net, Matrice* input_data, Matrice* output_data);

// Fonction pour entrainer ensemble des images
void train_batch_imgs(NeuralNetwork* net, uint8_t* images, uint8_t* labels, int size);

// Fonction de prediction d'une image par notre reseau
Matrice* predict_network(NeuralNetwork* net, Matrice* IMG);

// Fonction qui retourne le pourcentage des images correctes par notre reseau 
double predict_rate_network(NeuralNetwork* net, uint8_t* images, uint8_t* labels, int size);

// Fonction qui affiche les informations d'un reseau de neuron
void affiche_network(NeuralNetwork* net);

// Fonction qui libere la memoire de reseau
void free_network(NeuralNetwork *net);