struct neuron {
    float in_sum;
    float out;
};

struct layer {
    int n_neurons;
    struct neuron **neurons;
};

struct network {
    int n_layers;
    int n_neurons;
    float **biases;
    float ***weights;
    struct layer **layers;
};

struct network *create_network(int n_layers, int n_neurons[n_layers]);

void destroy_network(struct network *net);

void feedforward(struct network *net, float input[net->layers[0]->n_neurons],
             float output[net->layers[net->n_layers-1]->n_neurons]);

void network_set_random_weights_biases(struct network *net, float min, float max);

void network_set_weights(struct network *net, float *weights);

void network_get_weights(struct network *net, float *weights);

void network_set_biases(struct network *net, float biases[
              net->n_neurons - net->layers[0]->n_neurons]);

void network_get_biases(struct network *net, float biases[
              net->n_neurons - net->layers[0]->n_neurons]);

void calc_activs_deltas(struct network *net,
                    float input[net->layers[0]->n_neurons],
                    float output[net->layers[net->n_layers-1]->n_neurons],
                    float **activs, float **deltas);

void network_backprop(struct network *net, int batch_size,
                    int out_neurons, float input[][net->layers[0]->n_neurons],
                    float output[][out_neurons], float eta, int offset,
                    float ***activs, float ***deltas);

void network_update_minibatch(struct network *net, int batch_size,
                    float input[][net->layers[0]->n_neurons],
                    float output[][net->layers[net->n_layers-1]->n_neurons],
                    float eta, int offset);

void network_SGD(struct network *net, int train_size, int batch_size,
                 int epochs, float input[train_size][net->layers[0]->n_neurons],
                 float output[train_size][net->layers[net->n_layers-1]->
                                                                    n_neurons],
                 float eta, void fun(struct network *, int));

void shuffle(int len, int n_in, float inputs[len][n_in],
                      int n_out, float outputs[len][n_out]);

void exchange (int n_elems, float array1[n_elems], float array2[n_elems]);

float activation_function(float x);

float diff_activation_function(float x);

void diff_activation_function_vector(int n, float *in, float *out);

float cost_function_batch(int n, int n_tests, float output[][n],
                          float output_correct[][n]);

float cost_function(int n, float output[n], float output_correct[n]);

int network_save_to_file(struct network *net, char *filename);

int network_load_from_file(struct network *net, char *filename);

int __new_network_save_to_file(struct network *net, char *filename);

int __new_network_load_from_file(struct network *net, char *filename);
