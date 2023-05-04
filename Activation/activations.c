#include "activations.h"
#include <math.h>
#include "../Matrix/oper_mat.h"

float relu(float x){
    return (x > 0) ? x : 0;
}

Matrice* dRelu(Matrice* m) {
	double max = mat_max(m);
    Matrice* relu_deriv = create_mat(m->row, m->col);
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->col; j++) {
            relu_deriv->data[i * m->col + j] = (m->data[i * m->col + j] > 0) ? ((max - m->data[i * m->col + j])/(max*max)) : 0;
        }
    }
    return relu_deriv;
}

float sigmoid(float x) {
	return 1.0 / (1 + exp(-x));
}

Matrice* dSigmoid(Matrice* m) {
	Matrice* ones = create_mat(m->row, m->col);
	remplir_mat(ones, 1);
	Matrice* sigmoid_m = apply(sigmoid, m);
	Matrice* M1 = sub(ones, sigmoid_m);
	Matrice* M2 = mult(sigmoid_m, M1);
	free_mat(ones);
	free_mat(M1);
	free_mat(sigmoid_m);
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

Matrice* dSoftmax(Matrice* m) {
    Matrice* mat = create_mat(m->row, m->col);
    double sum = 0;
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->col; j++) {
            sum += exp(m->data[i * m->col + j]);
        }
    }
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->col; j++) {
            double s = exp(m->data[i * m->col + j]) / sum;
            mat->data[i * m->col + j] = s * (1 - s);
        }
    }
    return mat;
}

double init_weight() { 
	return ((1.0 * rand()) / (RAND_MAX /2)) - 1;  //des chiffres [-1 , 1]
}


double init_bias() {
	return (rand() % 20 + 1);
}


void Rand_Matrice(Matrice* m) {
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			double w = init_weight();
			m->data[i * m->col + j] = w;
		}
	}
}

void Rand_Bias(Matrice* m) {
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			double b = init_bias();
			m->data[i * m->col + j] = b;
		}
	}
}
