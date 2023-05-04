#pragma once
#include "matrice.h"




// Verifier les dimensions des matrices
int check_dimensions(Matrice *m1, Matrice *m2);

// Multiplication des matrices element par element
Matrice* mult(Matrice* m1, Matrice* m2);  

// Addition des matrices
Matrice* add(Matrice* m1, Matrice* m2);

// Soustraction des matrices
Matrice* sub(Matrice* m1, Matrice* m2);

// Appliquer une fonction sur une matrice
Matrice* apply(float (*func)(float), Matrice* m);

// Multiplication d'un scalaire par une matrice
Matrice* scale(double n, Matrice* m);

// Scalaire + Matrice
Matrice* addScalar(double n, Matrice* m);

// Transposé d'une matrice
Matrice* transpose(Matrice* m);

// Multiplication des matrices
Matrice* dotprod(Matrice* m1, Matrice* m2); 