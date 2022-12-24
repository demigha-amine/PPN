#include "matrice.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX 100

Matrice* create_mat(int row, int col) {
	Matrice *Matrice = malloc(sizeof(Matrice));
	Matrice->row = row;
	Matrice->col = col;
    Matrice->size = row * col ;
    Matrice->data = (double*) malloc((row*col) * sizeof(double) + 1);
	return Matrice;
}

void remplir_mat(Matrice *m, double n) {
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			m->data[i* m->row + j] = n;
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
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			printf("%1.3f ", m->data[i * m->row + j]);
		}
		printf("\n");
	}
}

Matrice* copy_mat(Matrice* m) {
	Matrice* mat = create_mat(m->row, m->col);
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[i * mat->row + j] = m->data[i * m->row + j];
		}
	}
	return mat;
}

void save_mat(Matrice* m, char* file) {
	FILE* fichier = fopen(file, "w");
	fprintf(fichier, "%d\n", m->row);
	fprintf(fichier, "%d\n", m->col);
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			fprintf(fichier, "%.6f\n", m->data[i * m->row + j]);
		}
	}
	printf("Successfully saved Matrice to %s\n", file);
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
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			fgets(tab, MAX, fichier);
			m->data[i * m->row + j] = strtod(tab, NULL);
		}
	}
	printf("Sucessfully loaded Matrice from %s\n", file);
	fclose(fichier);
	return m;
}

