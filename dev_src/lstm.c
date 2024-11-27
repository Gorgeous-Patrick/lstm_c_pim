#include "lstm.h"

static inline tensor ** create_tensor_array(int size){
    tensor ** array = (tensor **)SAFE_MALLOC(sizeof(tensor *) * size);
    return array;
}

LSTM * lstm_init(int input_size, int hidden_size, int output_size, int sequence_length){
    LSTM * lstm = (LSTM *)SAFE_MALLOC(sizeof(LSTM));

    int weight_shape[2] = {hidden_size, input_size};
    int output_shape[2] = {output_size, hidden_size};


    lstm->hidden_size = hidden_size;
    lstm->sequence_length = sequence_length;

    lstm->Wf = tensor_rand(weight_shape);
    lstm->Wi = tensor_rand(weight_shape);
    lstm->Wc = tensor_rand(weight_shape);
    lstm->Wo = tensor_rand(weight_shape);
    lstm->Wy = tensor_rand(output_shape);


    int init_shape[2] = {hidden_size, 1};
    lstm->hidden_states = create_tensor_array(lstm->sequence_length + 1);
    lstm->hidden_states[0] = tensor_zeros(init_shape);

    lstm->cell_states = create_tensor_array(lstm->sequence_length + 1);
    lstm->cell_states[0] = tensor_zeros(init_shape);

    lstm->concat_inputs = create_tensor_array(lstm->sequence_length);
    lstm->forget_gates = create_tensor_array(lstm->sequence_length);
    lstm->input_gates = create_tensor_array(lstm->sequence_length);
    lstm->candidate_gates = create_tensor_array(lstm->sequence_length);
    lstm->output_gates = create_tensor_array(lstm->sequence_length);

    lstm->outputs = create_tensor_array(lstm->sequence_length);

    return lstm;
}

tensor ** lstm_forward(LSTM * self, tensor * input){
    for(int i = 0; i < self->sequence_length; i++){
        self->concat_inputs[i] = tensor_concat(self->hidden_states[i], tensor_index(input, i));

        self->forget_gates[i] = _sigmoid(_mat_mul(self->Wf, self->concat_inputs[i]));
        self->input_gates[i] = _sigmoid(_mat_mul(self->Wi, self->concat_inputs[i]));
        self->candidate_gates[i] = _tanh(_mat_mul(self->Wc, self->concat_inputs[i]));
        self->output_gates[i] = _sigmoid(_mat_mul(self->Wo, self->concat_inputs[i]));

        self->cell_states[i + 1] = _mul(self->forget_gates[i], self->cell_states[i]);
        self->cell_states[i + 1] = _plus(self->cell_states[i + 1], _mul(self->input_gates[i], self->candidate_gates[i])); 

        self->hidden_states[i + 1] = _mul(self->output_gates[i], _tanh(self->cell_states[i+1]));

        self->outputs[i] = _mat_mul(self->Wy, self->hidden_states[i+1]);
    }

    return self->outputs;
}

void tensor_array_cleanup(tensor ** array, int size){
    for(int i = 0; i < size; i++){
        if(array[i]!=NULL){
            tensor_cleanup(array[i]);
        }
    }
}

void lstm_cleanup(LSTM * this){
    if(this == NULL){
        return;
    }

    tensor_cleanup(this->Wf);
    tensor_cleanup(this->Wi);
    tensor_cleanup(this->Wo);
    tensor_cleanup(this->Wy);

    tensor_array_cleanup(this->hidden_states, this->sequence_length);
    tensor_array_cleanup(this->cell_states, this->sequence_length);
    tensor_array_cleanup(this->concat_inputs, this->sequence_length);
    tensor_array_cleanup(this->forget_gates, this->sequence_length);
    tensor_array_cleanup(this->input_gates, this->sequence_length);
    tensor_array_cleanup(this->candidate_gates, this->sequence_length);
    tensor_array_cleanup(this->output_gates, this->sequence_length);
    tensor_array_cleanup(this->outputs, this->sequence_length);




    // free(this->hidden_states);
    // free(this->cell_states);

    // free(this);
}