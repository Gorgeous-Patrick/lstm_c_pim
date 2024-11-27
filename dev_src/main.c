#include <stdio.h>

#include "tensor.h"
#include "lstm.h"


int main(){
    int sequence_length = 15;
    int input_length = 32;
    int hidden_size = 25;


    int output_size = sequence_length;
    int input_size = hidden_size + sequence_length;

    LSTM * lstm = lstm_init(input_size, hidden_size, output_size, sequence_length);

    int input_shape[2] = {input_length, sequence_length};
    tensor * input = tensor_rand(input_shape);


    tensor ** outputs = lstm_forward(lstm, input);

    for(int i = 0; i < sequence_length; i++){
        tensor_printf(outputs[i]);
    }

    lstm_cleanup(lstm);
}