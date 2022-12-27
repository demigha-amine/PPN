#include "activations.h"
#include <math.h>
#include "../Matrix/oper_mat.h"



double sigmoid(double x) {
	return 1.0 / (1 + exp(-x));
}



Matrice* dSigmoid(Matrice* m) {
	Matrice* ones = create_mat(m->row, m->col);
	remplir_mat(ones, 1);
	Matrice* M1 = sub(ones, m);
	Matrice* M2 = mult(m, M1);
	free_mat(ones);
	free_mat(M1);
	return M2;
}

Matrice* softmax(Matrice* m) {
	double total = 0;
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			total += exp(m->data[i * m->col + j]);
		}
	}
	Matrice* mat = create_mat(m->row, m->col);
	for (int i = 0; i < mat->row; i++) {
		for (int j = 0; j < mat->col; j++) {
			mat->data[i * mat->col + j] = exp(m->data[i * m->col + j]) / total;
		}
	}
	return mat;
}


double init_weight() { 
	return ((double)rand()) / ((double)RAND_MAX); 
}

void Rand_Matrice(Matrice* m) {
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			double w = init_weight();
			m->data[i * m->col + j] = w;
		}
	}
	

}