#include <stdio.h>
#include <stdlib.h>
#include "matrice.h"
#include "oper_mat.h"



int main() {

    Matrice* M1 = create_mat(2, 2);
    remplir_mat(M1, 2);
    printf("create M1\n");
    affiche_mat(M1);

    Matrice* M2 = copy_mat(M1);
    printf("ccopy M1\n");
    affiche_mat(M2);

    Matrice* M3 = create_mat(2, 2);
    remplir_mat(M3, 1);
    printf("create M3\n");
    affiche_mat(M3);

    
    Matrice* M4 = sub(M1, M2);
    printf("M1 - M2\n");
    affiche_mat(M4);

    Matrice* M5 = add(M1, M3);
    printf("M1 + M3\n");
    affiche_mat(M5);

    Matrice* M6 = mult(M1, M3);
    printf("M1 * M3\n");
    affiche_mat(M6);

    
    Matrice* M7 = scale(1.5, M2);
    printf("M2 * 1.5\n");
    affiche_mat(M7);

    
    M2 = addScalar(2.5, M2);
    printf("2.5 + M2\n");
    affiche_mat(M2);

    
    M7 = transpose(M7);
    printf("TRANSP M7\n");
    affiche_mat(M7);

    Matrice* M8 = create_mat(2,3);
    remplir_mat(M8,3);

    Matrice* M11 = create_mat(2, 2);
    remplir_mat(M11, 2);

    Matrice* M9 = dotprod(M11,M8);
    printf("dot M9\n");
    affiche_mat(M9);


    free_mat(M1);
    free_mat(M2);
    free_mat(M3);
    free_mat(M4);
    free_mat(M5);
    free_mat(M6);
    free_mat(M7);

}