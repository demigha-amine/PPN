#pragma once
#include "../Activation/activations.h"
#include "../Matrix/oper_mat.h"


typedef struct {

	int Num_Inputs;		//Nbr des elements d'entrees
	int Num_Hidden;		//Nbr des noeuds caches
	int Num_Outputs;	//Nbr des elements de sortie
	double LR;			//Learning Rate


	Matrice* hidden_weights;		// Matrice des weights caches
	Matrice* output_weights;		// Matrice des weights de sortie
	Matrice* bias;					// Matrice des bias


} NeuralNetwork;


// Fonction de creation d'un reseau de neurones
NeuralNetwork* create_network(int input, int hidden, int output, double lr);

// Fonction pour entrainer le reseau de neurones
void train_network(NeuralNetwork* net, Matrice* input_data, Matrice* output_data);