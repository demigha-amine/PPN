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

		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				m->data[i * m1->col + j] +=  m1->data[i * m1->col + j] * m2->data[i * m1->col + j];
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch mult: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}




// Matrice* add(Matrice *m1, Matrice *m2) {
// 	if (check_dimensions(m1, m2)) {
// 		Matrice *m = create_mat(m1->row, m1->col);

// 		for (int i = 0; i < m1->row; i++) {
// 			for (int j = 0; j < m2->col; j++) {
// 				m->data[i * m1->col + j] +=  m1->data[i * m1->col + j] + m2->data[i * m2->col + j];
// 			}
// 		}
// 		return m;
// 	} else {
// 		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
// 		exit(1);
// 	}
// }

Matrice* add(Matrice* m1, Matrice* m2) {
    if (check_dimensions(m1, m2)) {
        Matrice* m = create_mat(m1->row, m1->col);
        cblas_saxpy(m1->row * m1->col, 1.0, m1->data, 1, m->data, 1);
        cblas_saxpy(m2->row * m2->col, 1.0, m2->data, 1, m->data, 1);
        return m;
    } else {
        printf("Dimension mismatch add: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
        exit(1);
    }
}


// Matrice* sub(Matrice *m1, Matrice *m2) {
// 	if (check_dimensions(m1, m2)) {
// 		Matrice *m = create_mat(m1->row, m1->col);

// 		for (int i = 0; i < m1->row; i++) {
// 			for (int j = 0; j < m2->col; j++) {
// 				m->data[i * m1->col + j] +=  m1->data[i * m1->col + j] - m2->data[i * m2->col + j];
// 			}
// 		}
// 		return m;
// 	} else {
// 		printf("Dimension mistmatch sub: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
// 		exit(1);
// 	}
// }

Matrice* sub(Matrice *m1, Matrice *m2) {
    if (check_dimensions(m1, m2)) {
        Matrice *m = create_mat(m1->row, m1->col);
        cblas_scopy(m1->row*m1->col, m1->data, 1, m->data, 1);
        cblas_saxpy(m2->row*m2->col, -1.0, m2->data, 1, m->data, 1);
        return m;
    } else {
        printf("Dimension mistmatch sub: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
        exit(1);
    }
}


Matrice* apply(float (*func)(float), Matrice* m) {
	Matrice *mat = copy_mat(m);

	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[i * m->col + j] = (*func)(m->data[i * m->col + j]);
		}
	}
	return mat;
}

// Matrice* scale(float n, Matrice* m) {
// 	Matrice* mat = copy_mat(m);

// 	for (int i = 0; i < m->row; i++) {
// 		for (int j = 0; j < m->col; j++) {
// 			mat->data[i * m->col + j] *= n;
// 		}
// 	}
// 	return mat;
// }

Matrice* scale(float n, Matrice* m) {
    Matrice* mat = create_mat(m->row, m->col);
    cblas_scopy(m->row * m->col, m->data, 1, mat->data, 1);
    cblas_sscal(m->row * m->col, n, mat->data, 1);
    return mat;
}


Matrice* addScalar(float n, Matrice* m) {
	Matrice* mat = copy_mat(m);

	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[i * m->col + j] += n;
		}
	}
	return mat;
}

Matrice* transpose(Matrice* m) {
	Matrice* mat = create_mat(m->col, m->row);

	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[j * mat->col + i] = m->data[i * m->col + j];
		}
	}
	return mat;
}


// Matrice* dotprod(Matrice* m1, Matrice* m2){
// 	if (m1->col == m2->row) {
// 		Matrice *m = create_mat(m1->row, m2->col);

// 		for (int i = 0; i < m1->row; i++) {
// 			for (int j = 0; j < m2->col; j++) {
// 				float sum = 0;

// 				for (int k = 0; k < m2->row; k++) {
// 					sum += m1->data[i * m1->col + k] * m2->data[k * m2->col + j];
// 				}
// 				m->data[i* m->col + j] = sum;
// 			}
// 		}
// 		return m;
// 	} else {
// 		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
// 		exit(1);
// 	}
// }


Matrice* dotprod(Matrice* m1, Matrice* m2) {
    if (m1->col == m2->row) {
        Matrice* m = create_mat(m1->row, m2->col);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1->row, m2->col, m1->col, 1.0, m1->data, m1->col, m2->data, m2->col, 0.0, m->data, m->col);
        return m;
    } else {
        printf("Dimension mismatch dot: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
        exit(1);
    }
}