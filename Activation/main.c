#include <stdio.h>
#include <stdlib.h>
#include "activations.h"



int main() {

    Matrice* M1 = create_mat(5, 5);
    remplir_mat(M1, 2);
    printf("create M1\n");
    affiche_mat(M1);

    double x = sigmoid(5);
    printf("x = %f\n", x);


    double w = init_weight();
    printf("w = %f\n", w);


    Matrice* M2 = dSigmoid(M1);
    printf("dSigmoid M1\n");
    affiche_mat(M2);


    Matrice* M3 = softmax(M1);
    printf("softmax M1\n");
    affiche_mat(M3);

    Rand_Matrice(M3);
    affiche_mat(M3);


    free_mat(M1);
    free_mat(M2);
    free_mat(M3);

}