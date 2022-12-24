#pragma once
#include <stdlib.h>
#include "../Matrix/matrice.h"


// Sigmoid fonction
double sigmoid(double x);

// Derive de Sigmoid
Matrice* dSigmoid(Matrice* m);

// Softmax probabilite fonction
Matrice* softmax(Matrice* m);

// Weights fonction
double init_weight();

// Creation d'une matrice random
void Rand_Matrice(Matrice* m);