#include "lstm.h"

static inline tensor ** create_tensor_array(int size){
    tensor ** array = (tensor **)SAFE_MALLOC(sizeof(tensor *) * size);
    return array;
}

static inline void allocate_tensor_memory(tensor ** array, int * shape, int start, int len){
    assert(start < len);

    for(int i = start; i < len; i++){
        array[i] = tensor_init(2, shape);
    }   
}

static inline tensor ** allocate_tensor_array(int size, int * shape){
    tensor ** array = create_tensor_array(size);
    allocate_tensor_memory(array, shape, 0, size);
    return array;
}

static inline void tensor_array_cleanup(tensor ** array, int size){
    for(int i = 0; i < size; i++){
        tensor_cleanup(array[i]);
    }

    SAFE_FREE(array);
}

LSTM * lstm_init(int input_size, int hidden_size, int output_size, int sequence_length){
    LSTM * lstm = (LSTM *)SAFE_MALLOC(sizeof(LSTM));

    int weight_shape[2] = {hidden_size, input_size};
    int output_shape[2] = {output_size, hidden_size};


    lstm->hidden_size = hidden_size;
    lstm->sequence_length = sequence_length;

    lstm->Wf = tensor_rand_(weight_shape);
    lstm->Wi = tensor_rand_(weight_shape);
    lstm->Wc = tensor_rand_(weight_shape);
    lstm->Wo = tensor_rand_(weight_shape);
    lstm->Wy = tensor_rand_(output_shape);


    int init_shape[2] = {hidden_size, 1};
    lstm->hidden_states = create_tensor_array(lstm->sequence_length + 1);
    lstm->hidden_states[0] = tensor_zeros(init_shape);
    allocate_tensor_memory(lstm->hidden_states, init_shape, 1, lstm->sequence_length + 1);

    lstm->cell_states = create_tensor_array(lstm->sequence_length + 1);
    lstm->cell_states[0] = tensor_zeros(init_shape);
    allocate_tensor_memory(lstm->cell_states, init_shape, 1, lstm->sequence_length + 1);

    int concat_shape[2] = {input_size, 1};
    lstm->concat_inputs = allocate_tensor_array(lstm->sequence_length, concat_shape);


    lstm->forget_gates = allocate_tensor_array(lstm->sequence_length, init_shape);
    lstm->input_gates = allocate_tensor_array(lstm->sequence_length, init_shape);
    lstm->candidate_gates = allocate_tensor_array(lstm->sequence_length, init_shape);
    lstm->output_gates = allocate_tensor_array(lstm->sequence_length, init_shape);

    lstm->outputs = allocate_tensor_array(lstm->sequence_length, init_shape);

    return lstm;
}

tensor ** lstm_forward(LSTM * self, tensor * input){
    int * input_shape = tensor_shape(input);

    for(int i = 0; i < self->sequence_length; i++){
        tensor * subtensor = tensor_init(2, input_shape);
        
        tensor_select(subtensor, input, i);
        tensor_concat(self->concat_inputs[i], self->hidden_states[i], subtensor);

        tensor_mat_mul(self->forget_gates[i], self->Wf, self->concat_inputs[i]);
        tensor_sigmoid_(self->forget_gates[i]);

        tensor_mat_mul(self->input_gates[i], self->Wi, self->concat_inputs[i]);
        tensor_sigmoid_(self->input_gates[i]);

        tensor_mat_mul(self->candidate_gates[i], self->Wc, self->concat_inputs[i]);
        tensor_tanh_(self->candidate_gates[i]);

        tensor_mat_mul(self->output_gates[i], self->Wo, self->concat_inputs[i]);
        tensor_sigmoid_(self->output_gates[i]);

        tensor_mul(self->cell_states[i], self->forget_gates[i], self->cell_states[i]);
        tensor_mul(self->cell_states[i + 1], self->input_gates[i], self->candidate_gates[i]);
        tensor_plus(self->cell_states[i + 1], self->cell_states[i + 1], self->cell_states[i]);

        tensor_tanh(self->hidden_states[i + 1], self->cell_states[i + 1]);
        tensor_mul(self->hidden_states[i + 1], self->output_gates[i], self->hidden_states[i + 1]);

        tensor_mat_mul(self->outputs[i], self->Wy, self->hidden_states[i+1]);

        tensor_cleanup(subtensor);
    }

    return self->outputs;
}



void lstm_cleanup(LSTM * this){
    if(this == NULL){
        return;
    }

    tensor_cleanup(this->Wf);
    tensor_cleanup(this->Wi);
    tensor_cleanup(this->Wo);
    tensor_cleanup(this->Wy);
    tensor_cleanup(this->Wc);

    tensor_array_cleanup(this->hidden_states, this->sequence_length + 1);
    tensor_array_cleanup(this->cell_states, this->sequence_length + 1);

    tensor_array_cleanup(this->concat_inputs, this->sequence_length);
    tensor_array_cleanup(this->forget_gates, this->sequence_length);
    tensor_array_cleanup(this->input_gates, this->sequence_length);
    tensor_array_cleanup(this->candidate_gates, this->sequence_length);
    tensor_array_cleanup(this->output_gates, this->sequence_length);
    tensor_array_cleanup(this->outputs, this->sequence_length);

    SAFE_FREE(this);
}