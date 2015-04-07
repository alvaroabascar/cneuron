#include <matrix.h>

/**
 * @file neuron.h
 * @author Alvaro Abella Bascaran
 * @date 7 apr 2015
 * @brief Interface to cNeuron.
 *
 * @see https://github.com/alvaroabascar/cneuron.git
 */
struct network {
   /** Number of layers of the network, including input and output layers.*/
    int n_layers;
    /** Array of matrices for the biases. Each matrix will actually contain
     * a single column, with biases[i][k] containing the bias of the kth
     * neuron in layer "i".
     */
    matrix_double *biases;
    /** Array of matrices for the weights. weights[i][j][k] will contain
     * the weight which links neuron "k" in layer "i" with neuron "j" in
     * layer "i+1"
     */
    matrix_double *weights;
    /** Array indicating the number of neurons in each layer.
     * Eg. for a net with 1, 20 and 10 neurons, net_structure = {1, 20, 10}
     */
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
