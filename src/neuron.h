#include <matrix.h>

struct network {
    int n_layers;
    matrix_double *biases;
    matrix_double *weights;
    int *net_structure;
};

struct network create_network(int n_layers, int *n_neurons);

void destroy_network(struct network net);

void set_random_weights_biases(struct network net);

void save_network(struct network net, char *filename);

struct network load_network(char *filename);

void feedforward(struct network net, double *input, double *output);

void network_SGD(struct network net, matrix_double training_data,
                 matrix_double training_labels, int epochs,
                 int mini_batch_size, double eta);

void network_backprop(struct network net, matrix_double training_data,
                      matrix_double training_labels);

void shuffle_data(matrix_double data, matrix_double labels);

void vectorized_sigma(matrix_double matrix);

double sigma(double x);
