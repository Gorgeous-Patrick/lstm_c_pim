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
tensor * tensor_rand(int ndims, int shape[MAX_DIM]);

/**
 * @brief Concat two tensors t1 and t2, returning the results to self
 * 
 * @param self output tensor
 * @param t1 input tensor 1
 * @param t2 input tensor 2
 * @return self
 */
tensor * tensor_concat(tensor * self, tensor * t1, tensor * t2);

/**
 * @brief Concat tensors self and t1 and return result to self
 * 
 * @param self output tensor
 * @param t1 input tensor 1
 * @return self
 */
#define tensor_concat_(self, t1) tensor_concat(self, self, t1) 

/**
 * @brief add two tensors t1 and t2, returning the results to self
 * 
 * @param self output tensor
 * @param t1 input tensor 1
 * @param t2 input tensor 2
 * @return self
 */
tensor * tensor_plus(tensor * self, tensor * t1, tensor * t2);

/**
 * @brief add tensors self and t1 and return result to self
 * 
 * @param self output tensor
 * @param t1 input tensor 1
 * @return self
 */
#define tensor_plus_(self, t1) tensor_plus(self, self, t1)

/**
 * @brief apply hadamard on two tensors t1 and t2, returning the results to self
 * 
 * @param self output tensor
 * @param t1 input tensor 1
 * @param t2 input tensor 2
 * @return self
 */
tensor * tensor_mul(tensor * self, tensor * t1, tensor * t2);

/**
 * @brief apply hadamard on two tensors t1 and self, returning the results to self
 * 
 * @param self output tensor
 * @param t1 input tensor 1
 * @return self
 */
#define tensor_mul_(self, t1) tensor_mul(self, self, t1)


/**
 * @brief Creates a new subtensor with shallow reference to the data from the original tensor
 * 
 * @param self output tensor
 * @param t1 input tensor 1
 * @return self
 */
tensor * tensor_select(tensor * self, tensor * src, int index);

/**
 * @brief multiply two tensors t1 and t2, returning the results to self
 * 
 * @param self output tensor
 * @param t1 input tensor 1
 * @param t2 input tensor 2
 * @return self
 */
tensor * tensor_mat_mul(tensor * self, tensor * t1, tensor * t2);

/*
Create a deep copy of the tensor
*/
void tensor_clone(tensor * self, tensor * src);



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
#define tensor_rand_(shape) tensor_create(tensor_rand, shape)


void tensor_printf(tensor * self);
void tensor_cleanup(tensor * self);

#endif // TENSOR_H