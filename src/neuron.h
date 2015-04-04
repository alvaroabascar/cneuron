#include <matrix.h>

struct network {
    int n_layers;
    matrix_double *biases;
    matrix_double *weights;
    int *net_structure;
};

struct network create_network(int n_layers, int *n_neurons);

void destroy_network(struct network net);

void network_set_random_weights_biases(struct network net);

void feedforward(struct network net, double *input, double *output);

void vectorized_sigma(matrix_double matrix);

double sigma(double x);
