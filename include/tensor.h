#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#define MAX_DIM 2

typedef struct tensor_data tensor_data;
typedef struct tensor tensor;

tensor * tensor_init(int ndims, int shape[MAX_DIM]);
tensor * _tensor_zeros(int ndims, int shape[MAX_DIM]);
tensor * _tensor_ones(int ndims, int shape[MAX_DIM]);
tensor * _tensor_rand(int ndims, int shape[MAX_DIM]);

/*
Concat two tensors t1 and t2 and create a new tensor. Both t1 and t2 data is copied 
onto the new tensor. The caller is responsible for freeing the memory.
*/
tensor * tensor_concat(tensor * t1, tensor * t2);

/*
Adds two tensors t1 and t2 and create a new tensor for the result. 
The caller is responsible for freeing the memory of the new tensor.
*/
tensor * tensor_plus(tensor * t1, tensor * t2);

/*
Multiply two tensors t1 and t2 and create a new tensor for the result. 
The caller is responsible for freeing the memory of the new tensor.
*/
tensor * tensor_mul(tensor * t1, tensor * t2);

/*
Creates a new subtensor with shallow reference to the data from the original tensor
*/
tensor * tensor_index(const tensor * self, int index);

/*
Matrix multiplication between two tensors. 
*/
tensor * tensor_mat_mul(tensor * self, tensor * other);

/*
Create a deep copy of the tensor
*/
tensor * tensor_copy(tensor * self);

/*
Calculate the signmoid of every element in the array
*/
tensor * tensor_sigmoid(tensor * self);

tensor * tensor_tanh(tensor * self);

#define tensor_create(create, shape) create((ARRAY_LENGTH(shape)),shape)
#define tensor_zeros(shape) tensor_create((_ ## tensor_zeros), shape)
#define tensor_ones(shape) tensor_create((_ ## tensor_ones), shape)
#define tensor_rand(shape) tensor_create((_ ## tensor_rand), shape)


void tensor_printf(tensor * self);
void tensor_cleanup(tensor * self);

#endif // TENSOR_H