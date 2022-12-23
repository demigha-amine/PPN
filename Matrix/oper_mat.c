#include "oper_mat.h"
#include <stdlib.h>
#include <stdio.h>


int check_dimensions(Matrice *m1, Matrice *m2) {
	if (m1->row == m2->row && m1->col == m2->col) return 1;
	return 0;
}

Matrice* mult(Matrice *m1, Matrice *m2) {
	if (check_dimensions(m1, m2)) {
		Matrice *m = create_mat(m1->row, m1->col);
		for (int i = 0; i < m1->row; i++)
			{
			for (int k = 0; k < m2->col; k++)
			{
			const double _a_ = m1->data[i * m1->row + k];
			
			for (int j = 0; j < m2->row; j++)
				m->data[i * m1->row + j] +=  _a_ * m2->data[k * m1->row + j];
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch mult: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}

Matrice* add(Matrice *m1, Matrice *m2) {
	if (check_dimensions(m1, m2)) {
		Matrice *m = create_mat(m1->row, m1->col);
		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				m->data[i * m1->row + j] +=  m1->data[i * m1->row + j] + m2->data[i * m2->row + j];
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}

Matrice* sub(Matrice *m1, Matrice *m2) {
	if (check_dimensions(m1, m2)) {
		Matrice *m = create_mat(m1->row, m1->col);
		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				m->data[i * m1->row + j] +=  m1->data[i * m1->row + j] - m2->data[i * m2->row + j];
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch sub: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}

Matrice* apply(double (*func)(double), Matrice* m) {
	Matrice *mat = copy_mat(m);
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[i * m->row + j] = (*func)(m->data[i * m->row + j]);
		}
	}
	return mat;
}

Matrice* scale(double n, Matrice* m) {
	Matrice* mat = copy_mat(m);
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[i * m->row + j] *= n;
		}
	}
	return mat;
}

Matrice* addScalar(double n, Matrice* m) {
	Matrice* mat = copy_mat(m);
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[i * m->row + j] += n;
		}
	}
	return mat;
}

Matrice* transpose(Matrice* m) {
	Matrice* mat = create_mat(m->col, m->row);
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[j * m->col + i] = m->data[i * m->row + j];
		}
	}
	return mat;
}