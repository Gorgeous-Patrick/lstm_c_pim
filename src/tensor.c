#include <time.h>
#include <string.h>
#include <assert.h>

#include "tensor.h"

struct tensor_data{
    double * ptr;
    int size;
    struct ref refcount;
};

static void data_free(const struct ref *ref){
    tensor_data * data = container_of(ref, tensor_data, refcount);
    if(data){
        if(data->ptr){
            free(data->ptr);
            data->ptr = NULL;
        }
        free(data);
    }
    

    data = NULL;
}

tensor_data * data_init(int size){
    assert(size > 0);

    tensor_data * data = (tensor_data *)SAFE_MALLOC(sizeof(tensor_data));
    data->size = size;
    data->ptr = (double *)SAFE_MALLOC(sizeof(double) * size);
    data->refcount = (struct ref){.count = 1, .free = data_free};

    return data;
}

static inline void data_insert(tensor_data * self, double value, int index){
    assert(index < self->size);

    self->ptr[index] = value;
}

static inline double data_get(tensor_data * self, int index){
    assert(index < self->size);

    return self->ptr[index];
}

static inline const double * data_raw_ptr(tensor_data * self){
    return self->ptr;
}

static inline void data_assign_ptr(tensor_data * self, double * ptr){
    self->ptr = ptr;
}

static inline tensor_data * data_create_ref(tensor_data * src, int offset){
    ref_inc(&src->refcount);

    tensor_data * dest = data_init(src->size - offset);
    dest->refcount = src->refcount;
    dest->ptr = src->ptr + offset;
    return dest;
}

struct tensor{
    tensor_data * data;
    int shape[MAX_DIM];
    int ndims;
    int length;
};

static inline double _serial_dot_product(const double * a, const double * b, unsigned int length){
    double output = 0;
    for(size_t i = 0; i < length; i++){
        output += a[i] * b[i];
    }

    return output;
}

static inline double * _serial_matrix_multiplication(const double * a, const double * b, int m, int p, int n){
    double * c = (double *)SAFE_MALLOC(sizeof(double) * m * p);
    
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

static inline double * _serial_addition(const double * vec1, const double * vec2, unsigned int length){
    double * result = (double *)SAFE_MALLOC(sizeof(double) * length);

    for(size_t i = 0; i < length; i++){
        result[i] = vec1[i] + vec2[i];
    }

    return result;
}

static inline double * _serial_multiplication(const double * vec1, const double * vec2, unsigned int length){
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
    t->data = NULL;

    return t;
}

tensor * tensor_init(int ndims, int shape[MAX_DIM]){
    tensor * t = _tensor_shallow_init(ndims, shape);
    t->data = data_init(t->length);

    return t;
}

tensor * _tensor_zeros(int ndims, int shape[MAX_DIM]){
    tensor * t = tensor_init(ndims, shape);

    for(int i = 0; i < t->length; i++){
        data_insert(t->data, 0.0, i);                
    }

    return t;
}

tensor * _tensor_ones(int ndims, int shape[MAX_DIM]){
    tensor * t = tensor_init(ndims, shape);

    for(int i = 0; i < t->length; i++){
        data_insert(t->data, 1, i);                 
    }

    return t;
}

/*
Generates a random number between -1 and 1
 */
tensor * _tensor_rand(int ndims, int shape[MAX_DIM]){
    tensor * t = tensor_init(ndims, shape);

    for(int i = 0; i < t->length; i++){
        double rand = (double)( 2 * arc4random_uniform(RAND_MAX))/RAND_MAX - 1;
        data_insert(t->data, rand, i);               
    }

    return t;
}

tensor * tensor_concat(tensor * t1, tensor * t2){
    if(t1->shape[1] != t2->shape[1]){
        PANIC("Tensor size mismatch");
    }



    int shape[2] = {(t1->shape[0] + t2->shape[0]), t1->shape[1]};

    tensor * result = tensor_init(2, shape);

    memcpy(result->data->ptr, t1->data->ptr, t1->length * sizeof(double));
    memcpy(result->data->ptr + t1->length, t2->data->ptr, t2->length * sizeof(double));

    return result;
}

tensor * tensor_binary_point_wise_op(tensor * t1, tensor * t2, double * (*op)(const double *, const double *, unsigned int)){
    if((t1->shape[0] != t2->shape[0]) || (t1->shape[1] != t2->shape[1])){
        PANIC("Tensor size mismatch");
    }

    tensor * result = tensor_init(2, t1->shape);
    data_assign_ptr(result->data, op(data_raw_ptr(t1->data), data_raw_ptr(t2->data), t1->length));

    return result;   
}

tensor * tensor_plus(tensor * t1, tensor * t2){
    return tensor_binary_point_wise_op(t1, t2, _serial_addition);
}

tensor * tensor_mul(tensor * t1, tensor * t2){
    return tensor_binary_point_wise_op(t1, t2, _serial_multiplication);
}

tensor * tensor_index(const tensor * self, int index){
    if(index >= self->shape[0]){
        PANIC("Index out of bounds");
    }

    int shape[2] = {self->shape[1], 1};
    tensor * result = _tensor_shallow_init(2, shape);

    //create a reference to the existing data
    result->data = data_create_ref(self->data, index * self->shape[1]);


    return result;
}


void tensor_printf(tensor * self){
    printf("Tensor(");
    printf("[");
    for(int i = 0; i < self->shape[0]; i++){
        printf("[");
        for(int j = 0; j < self->shape[1]; j++){
            int index = j + (i * self->shape[1]);
            printf("%g", data_get(self->data, index));

            if(j < self->shape[1] - 1){
                printf(",");
            }
        }
        printf("]");

        if(i < self->shape[0] - 1){
            printf(",");
        }
    }
    printf("]");
    printf(")\n");
}



tensor * tensor_mat_mul(tensor * self, tensor * other){
    if(self->shape[1] != other->shape[0]){
        PANIC("Mismatch tensor sizes [%d, %d] x [%d, %d]\n", self->shape[0], self->shape[1], other->shape[0], other->shape[1]);
    }

    int new_shape[2] = {self->shape[0], other->shape[1]};
    tensor * result = tensor_init(2, new_shape);

    if(new_shape[0] == 1 && new_shape[1] == 1){
        data_insert(
            result->data, 
            _serial_dot_product(data_raw_ptr(self->data), data_raw_ptr(other->data), self->shape[1]), 
            0
        );
    }else{
        data_assign_ptr(result->data,
            _serial_matrix_multiplication(
                data_raw_ptr(self->data), 
                data_raw_ptr(other->data), 
                self->shape[0], 
                self->shape[1], 
                other->shape[1]
        ));
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
        data_insert(result->data, sigmoid(data_get(result->data, i)), i);
    }

    return result;
}

tensor * tensor_tanh(tensor * self){
    tensor * result = tensor_copy(self);

    for(int i = 0; i < self->length; i++){
        data_insert(result->data, tanh(data_get(result->data, i)), i);
    }

    return result;
}

void tensor_cleanup(tensor * self){
    if(self == NULL){
        return;
    }

    if(self->data != NULL){
        ref_dec(&self->data->refcount);
    }
    
    free(self);
}