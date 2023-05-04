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

		 //#pragma omp parallel for
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

// 		 //#pragma omp parallel for
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
        cblas_daxpy(m1->row * m1->col, 1.0, m1->data, 1, m->data, 1);
        cblas_daxpy(m2->row * m2->col, 1.0, m2->data, 1, m->data, 1);
        return m;
    } else {
        printf("Dimension mismatch add: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
        exit(1);
    }
}


// Matrice* sub(Matrice *m1, Matrice *m2) {
// 	if (check_dimensions(m1, m2)) {
// 		Matrice *m = create_mat(m1->row, m1->col);

// 		 //#pragma omp parallel for
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
        cblas_dcopy(m1->row*m1->col, m1->data, 1, m->data, 1);
        cblas_daxpy(m2->row*m2->col, -1.0, m2->data, 1, m->data, 1);
        return m;
    } else {
        printf("Dimension mistmatch sub: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
        exit(1);
    }
}


Matrice* apply(double (*func)(double), Matrice* m) {
	Matrice *mat = copy_mat(m);

	 //#pragma omp parallel for
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[i * m->col + j] = (*func)(m->data[i * m->col + j]);
		}
	}
	return mat;
}

// Matrice* scale(double n, Matrice* m) {
// 	Matrice* mat = copy_mat(m);

// 	 //#pragma omp parallel for
// 	for (int i = 0; i < m->row; i++) {
// 		for (int j = 0; j < m->col; j++) {
// 			mat->data[i * m->col + j] *= n;
// 		}
// 	}
// 	return mat;
// }

Matrice* scale(double n, Matrice* m) {
    Matrice* mat = create_mat(m->row, m->col);
    cblas_dcopy(m->row * m->col, m->data, 1, mat->data, 1);
    cblas_dscal(m->row * m->col, n, mat->data, 1);
    return mat;
}


Matrice* addScalar(double n, Matrice* m) {
	Matrice* mat = copy_mat(m);

	 //#pragma omp parallel for
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[i * m->col + j] += n;
		}
	}
	return mat;
}

Matrice* transpose(Matrice* m) {
	Matrice* mat = create_mat(m->col, m->row);

	 //#pragma omp parallel for
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			mat->data[j * mat->col + i] = m->data[i * m->col + j];
		}
	}
	return mat;
}





// Matrice* transpose(Matrice* m) {
//     Matrice* mat = create_mat(m->col, m->row);
//     cblas_transpose(CblasRowMajor, m->col, m->row, 1.0, m->data, m->col, mat->data, mat->col);
//     return mat;
// }




// Matrice* dotprod(Matrice* m1, Matrice* m2){
// 	if (m1->col == m2->row) {
// 		Matrice *m = create_mat(m1->row, m2->col);

// 		//#pragma omp parallel for
// 		for (int i = 0; i < m1->row; i++) {
// 			for (int j = 0; j < m2->col; j++) {
// 				double sum = 0;

// 				//#pragma omp parallel for reduction(+:sum)
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
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1->row, m2->col, m1->col, 1.0, m1->data, m1->col, m2->data, m2->col, 0.0, m->data, m->col);
        return m;
    } else {
        printf("Dimension mismatch dot: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
        exit(1);
    }
}