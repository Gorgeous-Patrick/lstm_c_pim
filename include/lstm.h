#ifndef LSTM_H
#define LSTM_H

#include "tensor.h"

#define _sigmoid tensor_sigmoid
#define _tanh tensor_tanh
#define _plus tensor_plus
#define _mul tensor_mul
#define _mat_mul tensor_mat_mul



typedef struct lstm{
    //hyperparameters
    int hidden_size;
    int sequence_length;

    //forget gate
    tensor * Wf;

    //input gate
    tensor * Wi;

    //output gate
    tensor * Wo;

    //cell state
    tensor * Wc;

    //output
    tensor * Wy;

    // These should be array lists instead of regular arrays
    tensor ** hidden_states;
    tensor ** cell_states;
    tensor ** concat_inputs;
    tensor ** forget_gates;
    tensor ** input_gates;
    tensor ** candidate_gates;
    tensor ** output_gates;

    tensor ** outputs;
} LSTM;

LSTM * lstm_init(int input_size, int hidden_size, int output_size, int sequence_length);
tensor ** lstm_forward(LSTM * lstm, tensor * input);
void lstm_cleanup(LSTM * this);

#endif // LSTM_H