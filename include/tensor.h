#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "data.h"

#define MAX_DIM 2

#define TENSOR_CHECK(cond, ...) if((cond)){PANIC(__VA_ARGS__);}
#define TENSOR_EXIST(t) TENSOR_CHECK((t == NULL), "Tensor undefined")

typedef struct tensor tensor;

tensor * tensor_init(int ndims, int shape[MAX_DIM]);
tensor * _tensor_zeros(int ndims, int shape[MAX_DIM]);
tensor * _tensor_ones(int ndims, int shape[MAX_DIM]);
tensor * _tensor_rand(int ndims, int shape[MAX_DIM]);

/*
Concat two tensors t1 and t2 and create a new tensor. Both t1 and t2 data is copied 
onto the new tensor. The caller is responsible for freeing the memory.
*/
tensor * tensor_concat(tensor * self, tensor * t1, tensor * t2);

/*
Adds two tensors t1 and t2 and create a new tensor for the result. 
The caller is responsible for freeing the memory of the new tensor.
*/
tensor * tensor_plus(tensor * self, tensor * t1, tensor * t2);
#define tensor_plus_(r, t) tensor_plus(r, r, t);
/*
Multiply two tensors t1 and t2 and create a new tensor for the result. 
The caller is responsible for freeing the memory of the new tensor.
*/
tensor * tensor_mul(tensor * self, tensor * t1, tensor * t2);

/*
Creates a new subtensor with shallow reference to the data from the original tensor
*/
tensor * tensor_select(tensor * self, tensor * src, int index);

/*
Matrix multiplication between two tensors. 
*/
tensor * tensor_mat_mul(tensor * self, tensor * t1, tensor * t2);

/*
Create a deep copy of the tensor
*/
tensor * tensor_copy(tensor * self);
void tensor_clone(tensor * self, tensor * src);

void tensor_copy_from_data(tensor * self, Data * data, int shape[], int ndims, int length, int offset);

/*
Calculate the signmoid of every element in the array and assigns the result to the out tensor
*/
void tensor_sigmoid(tensor * self, tensor * in);

int * tensor_shape(tensor * self);

#define tensor_sigmoid_(t) tensor_sigmoid(t, t)

/**
 * Get the raw pointer values pointed to by the tensor
 */
double * tensor_data(tensor * self);

void tensor_tanh(tensor * self, tensor * in);
#define tensor_tanh_(t) tensor_tanh(t, t)

#define tensor_create(create, shape) create((ARRAY_LENGTH(shape)),shape)
#define tensor_zeros(shape) tensor_create((_ ## tensor_zeros), shape)
#define tensor_ones(shape) tensor_create((_ ## tensor_ones), shape)
#define tensor_rand(shape) tensor_create((_ ## tensor_rand), shape)


void tensor_printf(tensor * self);
void tensor_cleanup(tensor * self);

#endif // TENSOR_H