#ifndef MAT_OPS_H
#define MAT_OPS_H

#include <stdlib.h>
#include "utils.h"

static inline void _serial_sigmoid(double * a, int length){
    for(int i = 0; i < length; i++){
        a[i] = sigmoid(a[i]);
    }
}

static inline void _serial_tanh(double * a, int length){
    for(int i = 0; i < length; i++){
        a[i] = tanh(a[i]);
    }
}

static inline double _serial_dot_product(const double * a, const double * b, unsigned int length){
    double output = 0;
    for(size_t i = 0; i < length; i++){
        output += a[i] * b[i];
    }

    return output;
}

static inline double * _serial_matrix_multiplication(const double * a, const double * b, double * c, int m, int p, int n){    
    // Initialize the result matrix to zero
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i * n + j] = 0.0;
        }
    }

    // Perform the matrix multiplication
    for (int i = 0; i < m; ++i) {          // Iterate over rows of mat1
        for (int j = 0; j < n; ++j) {      // Iterate over columns of mat2
            for (int k = 0; k < p; ++k) {  // Iterate over the common dimension
                c[i * n + j] += a[i * p + k] * b[k * n + j];
            }
        }
    }

    return c;
}

static inline void _serial_addition(const double * a, const double * b, double * c, unsigned int length){
    for(size_t i = 0; i < length; i++){
        c[i] = a[i] + b[i];
    }
}

static inline void _serial_multiplication(const double * a, const double * b, double * c, unsigned int length){
    for(size_t i = 0; i < length; i++){
        c[i] = a[i] * b[i];
    }
}

#endif // MAT_OPS_H