#pragma once
#include <stdlib.h>
#include "../Matrix/matrice.h"
// ReLu function
float relu(float x);

// ReLu derived
Matrice* dRelu(Matrice* m);

// Sigmoid fonction
float sigmoid(float x);

// Derive de Sigmoid
Matrice* dSigmoid(Matrice* m);

// Softmax probabilite fonction
Matrice* softmax(Matrice* m);

// Derive de Softmax
Matrice* dSoftmax(Matrice* m);

// Weights fonction
double init_weight();

// Bias fonction
double init_bias();

// Creation d'une matrice random
void Rand_Matrice(Matrice* m);

// Creation d'une matrice random pour les bias
void Rand_Bias(Matrice* m);
