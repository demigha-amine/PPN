#pragma once

//definition d'une matrice
typedef struct {
	double* data; //tableau de donnes
	int row; //nbr de lignes
	int col; //nbr de colonnes
	int size; //taille de tableau matrice taille = row * col
} Matrice;

//Creation d'une matrice
Matrice* create_mat(int row, int col);


//remplire la matrice
void remplir_mat(Matrice *m, int n);

//liberer la matrice
void free_mat(Matrice *m);

//Afficher la matrice
void affiche_mat(Matrice *m);

//retourne une copie de la matrice
Matrice* copy_mat(Matrice *m);

//sauvgarder une matrice sur un fichier
void save_mat(Matrice* m, char* fichier);

// charger une matrice d'un fichier
Matrice* charger_mat(char* fichier);


