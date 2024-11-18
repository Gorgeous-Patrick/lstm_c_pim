#include <time.h>
#include <string.h>

#include "tensor.h"


struct tensor{
    double * data;
    int shape[MAX_DIM];
    int ndims;
    int length;
};

static inline double _serial_dot_product(double * a, double * b, unsigned int length){
    double output = 0;
    for(size_t i = 0; i < length; i++){
        output += a[i] * b[i];
    }

    return output;
}

static inline void _serial_matrix_multiplication(double * a, double * b, double * c, int m, int p, int n){
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
}

static inline double * _serial_addition(double * vec1, double * vec2, unsigned int length){
    double * result = (double *)SAFE_MALLOC(sizeof(double) * length);

    for(size_t i = 0; i < length; i++){
        result[i] = vec1[i] + vec2[i];
    }

    return result;
}

static inline double * _serial_multiplication(double * vec1, double * vec2, unsigned int length){
    double * result = (double *)SAFE_MALLOC(sizeof(double) * length);

    for(size_t i = 0; i < length; i++){
        result[i] = vec1[i] * vec2[i];
    }

    return result;
}

static inline void _dim_check(int ndims){
    if(ndims < 1 || ndims > MAX_DIM){
        PANIC("Dimensions should be at least 1 and not greater than %d\n", MAX_DIM);
    }
}

static inline void _shape_check(int axis_size){
    if(axis_size < 1){
        PANIC("Axis length should at least be 1\n");
    }
}

static inline void _size_check(int ndims, int shape[MAX_DIM]){
    _dim_check(ndims);

    for(int i = 0; i < ndims; i++){
        _shape_check(shape[i]);
    }
}

static inline int _get_length(int ndims, int shape[MAX_DIM]){
    int length = 1;
    for(int i = 0; i < ndims; i++){
        length *= shape[i];
    }

    return length;
}

static inline tensor * _tensor_shallow_init(int ndims, int shape[MAX_DIM]){
    _size_check(ndims, shape);

    tensor * t = (tensor *)SAFE_MALLOC(sizeof(tensor));
    t->ndims = ndims;

    for(int i = 0; i < ndims; i++){
        t->shape[i] = shape[i];
    }

    t->length = _get_length(ndims, shape);

    return t;
}

tensor * tensor_init(int ndims, int shape[MAX_DIM]){
    tensor * t = _tensor_shallow_init(ndims, shape);
    t->data = (double *)SAFE_MALLOC(sizeof(double) * t->length);

    return t;
}

tensor * _tensor_zeros(int ndims, int shape[MAX_DIM]){
    tensor * t = tensor_init(ndims, shape);

    for(int i = 0; i < t->length; i++){
        t->data[i] = 0.0;                 
    }

    return t;
}

tensor * _tensor_ones(int ndims, int shape[MAX_DIM]){
    tensor * t = tensor_init(ndims, shape);

    for(int i = 0; i < t->length; i++){
        t->data[i] = 1;                 
    }

    return t;
}

/*
Generates a random number between -1 and 1
 */
tensor * _tensor_rand(int ndims, int shape[MAX_DIM]){
    tensor * t = tensor_init(ndims, shape);

    for(int i = 0; i < t->length; i++){
        t->data[i] = (double)(arc4random_uniform(RAND_MAX) + RAND_MAX)/RAND_MAX - 1;               
    }

    return t;
}

tensor * tensor_concat(tensor * t1, tensor * t2){
    if(t1->shape[1] != t2->shape[1]){
        PANIC("Tensor size mismatch");
    }

    int shape[2] = {(t1->shape[0] + t2->shape[0]), t1->shape[1]};

    tensor * result = tensor_init(2, shape);

    memcpy(result->data, t1->data, t1->length * sizeof(double));
    memcpy(result->data + t1->length, t2->data, t2->length * sizeof(double));

    return result;
}

tensor * tensor_point_wise_op(tensor * t1, tensor * t2, double * (*op)(double *, double *, unsigned int)){
    if((t1->shape[0] != t2->shape[0]) || (t1->shape[1] != t2->shape[1])){
        PANIC("Tensor size mismatch");
    }

    tensor * result = tensor_init(2, t1->shape);
    result->data = op(t1->data, t2->data, t1->length);

    return result;   
}

tensor * tensor_plus(tensor * t1, tensor * t2){
    return tensor_point_wise_op(t1, t2, _serial_addition);
}

tensor * tensor_mul(tensor * t1, tensor * t2){
    return tensor_point_wise_op(t1, t2, _serial_multiplication);
}

tensor * tensor_index(tensor * self, int index){
    if(index >= self->shape[0]){
        PANIC("Index out of bounds");
    }

    int shape[2] = {self->shape[1], 1};
    tensor * result = _tensor_shallow_init(2, shape);

    //create a reference to the existing data
    result->data = self->data + (index * self->shape[1]);


    return result;
}


void tensor_printf(tensor * self){
    printf("Tensor(");
    printf("[");
    for(int i = 0; i < self->shape[0]; i++){
        printf("[");
        for(int j = 0; j < self->shape[1]; j++){
            int index = j + (i * self->shape[1]);
            printf("%g", self->data[index]);

            if(j < self->shape[1] - 1){
                printf(",");
            }
        }
        printf("]");

        if(i < self->shape[0] - 1){
            printf(",");
        }
    }
    printf("]\n");
    printf(")\n");
}



tensor * tensor_mat_mul(tensor * self, tensor * other){
    if(self->shape[1] != other->shape[0]){
        PANIC("Mismatch tensor sizes [%d, %d] x [%d, %d]\n", self->shape[0], self->shape[1], other->shape[0], other->shape[1]);
    }

    int new_shape[2] = {self->shape[0], other->shape[1]};
    tensor * result = tensor_init(2, new_shape);

    if(new_shape[0] == 1 && new_shape[1] == 1){
        result->data[0] = _serial_dot_product(self->data, other->data, self->shape[1]);
    }else{
        _serial_matrix_multiplication(self->data, other->data, result->data, self->shape[0], self->shape[1], other->shape[1]);
    }


    return result;
}

tensor * tensor_copy(tensor * self){
    tensor * result = tensor_init(self->ndims, self->shape);
    memcpy(result->data, self->data, self->length * sizeof(double));

    return result;
}

tensor * tensor_sigmoid(tensor * self){
    tensor * result = tensor_copy(self);

    for(int i = 0; i < self->length; i++){
        result->data[i] = sigmoid(result->data[i]);
    }

    return result;
}

tensor * tensor_tanh(tensor * self){
    tensor * result = tensor_copy(self);

    for(int i = 0; i < self->length; i++){
        result->data[i] = tanh(result->data[i]);
    }

    return result;
}

void tensor_cleanup(tensor * self){
    if(self == NULL){
        return;
    }

    free(self->data);
    free(self);
}