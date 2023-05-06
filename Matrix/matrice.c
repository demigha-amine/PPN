#include "matrice.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX 100

Matrice* create_mat(int row, int col) {
	Matrice *matrice = malloc(sizeof(Matrice));
	matrice->row = row;
	matrice->col = col;
    matrice->size = row * col ;
    matrice->data = (float*) calloc((row*col) , sizeof(float));
	return matrice;
}

void remplir_mat(Matrice *m, float n) {

	#pragma omp parallel for

	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			m->data[i* m->col + j] = n;
		}
	}
}

void free_mat(Matrice *m) {
	free(m->data);
	free(m);
	m = NULL;
}

void affiche_mat(Matrice* m) {
	printf("Lignes = %d Columns = %d\n", m->row, m->col);

	#pragma omp parallel for

	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			printf("%1.3f ", m->data[i * m->col + j]);
		}
		printf("\n");
	}
}

// Matrice* copy_mat(Matrice* m) {
// 	Matrice* mat = create_mat(m->row, m->col);
// 	for (int i = 0; i < m->row; i++) {
// 		for (int j = 0; j < m->col; j++) {
// 			mat->data[i * mat->col + j] = m->data[i * m->col + j];
// 		}
// 	}
// 	return mat;
// }


Matrice* copy_mat(Matrice* m) {
    Matrice* mat = create_mat(m->row, m->col);
    cblas_scopy(m->row * m->col, m->data, 1, mat->data, 1);
    return mat;
}

void save_mat(Matrice* m, char* file) {
	FILE* fichier = fopen(file, "w");
	fprintf(fichier, "%d\n", m->row);
	fprintf(fichier, "%d\n", m->col);

	#pragma omp parallel for

	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			fprintf(fichier, "%.6f\n", m->data[i * m->col + j]);
		}
	}
	//printf("Successfully saved Matrice to %s\n", file);
	fclose(fichier);
}

Matrice* charger_mat(char* file) {
	FILE* fichier = fopen(file, "r");
	char tab[MAX]; 
	fgets(tab, MAX, fichier);
	int row = atoi(tab);

	fgets(tab, MAX, fichier);
	int col = atoi(tab);

	Matrice* m = create_mat(row, col);

	//#pragma omp parallel for

	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			fgets(tab, MAX, fichier);
			m->data[i * m->col + j] = strtod(tab, NULL); 
		}
	}
	printf("Sucessfully loaded Matrice from %s\n", file);
	fclose(fichier);
	return m;
}


int mat_argmax(Matrice* m) {
	// Pour les matrice vecteur (M*1)
	float max_arg = 0;
	int max_idx = 0;

	#pragma omp parallel for shared(max_arg,max_idx)

	for (int i = 0; i < m->row; i++) {
		if (m->data[i] > max_arg) {
			max_arg = m->data[i];
			max_idx = i;
		}
	}
	return max_idx;
}


float mat_max(Matrice* m) {
	// Pour les matrice vecteur (M*1)
	float max_arg = m->data[0];

	#pragma omp parallel for shared(max_arg)

	for (int i = 1; i < m->row; i++) {
		if (m->data[i] > max_arg) {
			max_arg = m->data[i];
		}
	}
	return max_arg;
}
