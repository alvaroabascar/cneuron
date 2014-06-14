#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include "neuron.h"
#include "matrix.h"

#define abs(x) ((x >= 0) ? (x) : (-1*(x)))

#define STOP_STEP 1e-10
#define STOP_COST 1e-2

struct network *create_network(int n_layers, int n_neurons[n_layers])
{
    struct network *net;
    struct layer *layer;
    struct neuron *neuron;
    int n_weights, n_biases;
    int i, n;

    /* allocate space for network and pointers to biases, weights, layers */
    net = malloc(sizeof(struct network));
    net->biases = malloc(n_layers * sizeof(float *));
    net->weights = malloc(n_layers * sizeof(float *));
    net->layers = malloc(n_layers * sizeof(struct layer *));
    net->n_layers = n_layers;
    net->n_neurons = 0;

    for (i = 0; i < n_layers; i++) {
        net->layers[i] = malloc(sizeof(struct layer));
        net->layers[i]->n_neurons = n_neurons[i];
        net->layers[i]->neurons = malloc(n_neurons[i] * sizeof(struct neuron));
        net->n_neurons += n_neurons[i];
        if (i > 0) {
            net->biases[i] = malloc(n_neurons[i] * sizeof(float));
            net->weights[i] = malloc(n_neurons[i-1] * sizeof(float *));
            for (n = 0; n < n_neurons[i-1]; n++)
                net->weights[i][n] = malloc(n_neurons[i] * sizeof(float));
        }
        for (n = 0; n < n_neurons[i]; n++) {
            net->layers[i]->neurons[n] = malloc(sizeof(struct neuron));
        }
    }
    network_set_random_weights_biases(net, -1.0, 1.0);
    return net;
}

void destroy_network(struct network *net)
{
    int l, n1, n2;
    for (l = 0; l < net->n_layers; l++) {
        for (n1 = 0; n1 < net->layers[l]->n_neurons; n1++) {
            free(net->layers[l]->neurons[n1]);
            if (l < net->n_layers-1) {
                free(net->weights[l+1][n1]);
            }
        }
        free(net->weights[l]);
        free(net->biases[l]);
        free(net->layers[l]);
    }
    free(net->weights);
    free(net->biases);
    free(net->layers);
    free(net);
}

/* feedforward:
 *      Input:
 *              net   -> a (trained) network
 *              input -> a vector of floats which is set as the input of 
 *                       the network. The length must be equal to the number
 *                       of neurons in the input layer.
 */
void feedforward(struct network *net, float input[net->layers[0]->n_neurons],
             float output[net->layers[net->n_layers-1]->n_neurons])
{
    int n1, n2, l;
    struct layer *layer, *layer_prev;
    struct neuron *neuron;
    /* Set input as the output from the input layer */
    for (n1 = 0; n1 < net->layers[0]->n_neurons; n1++)
        net->layers[0]->neurons[n1]->out = input[n1];
    
    /* For each layer (except the input layer)... */
    for (l = 1; l < net->n_layers; l++) {
        layer = net->layers[l];
        layer_prev = net->layers[l-1];
        /* For each neuron in the layer... */
        for (n2 = 0; n2 < layer->n_neurons; n2++) {
            neuron = layer->neurons[n2];
            neuron->in_sum = 0;
            /* For each neuron in the previous layer...*/
            for (n1 = 0; n1 < layer_prev->n_neurons; n1++)
                /* Add weighted activations from the previous layer */
                neuron->in_sum += layer_prev->neurons[n1]->out
                                  * net->weights[l][n1][n2];
            /* Add bias and compute the activation function */
            neuron->in_sum += net->biases[l][n2];
            neuron->out = activation_function(neuron->in_sum);
        }
    }
    /* Save network output into output array */
    for (n1 = 0; n1 < layer->n_neurons; n1++)
        output[n1] = layer->neurons[n1]->out;
}

/* network_set_random_weights_biases: Assign random weights to the network, uniformly
 *                             distributed between min and max
 * Input:
 *      net -> the network
 *      min -> minimum value of the weight
 *      max -> maximum value of the weight
 */
void network_set_random_weights_biases(struct network *net, float min,
                                       float max)
{
    int l, n1, n2;
    min = (min < 0) ? -1*min : min;
    for (l = 1; l < net->n_layers; l++) {
        for (n1 = 0; n1 < net->layers[l-1]->n_neurons; n1++)
            for (n2 = 0; n2 < net->layers[l]->n_neurons; n2++)
                net->weights[l][n1][n2] = (float)rand()/(float)RAND_MAX * (max+min) - min;
       for (n2 = 0; n2 < net->layers[l]->n_neurons; n2++)
           net->biases[l][n2] = (float)rand()/(float)RAND_MAX * (max+min) - min;
    }
}

void network_set_weights(struct network *net, float *weights)
{
    int l, n1, n2, i;
    i = 0;
    for (l = 1; l < net->n_layers; l++)
        for (n1 = 0; n1 < net->layers[l-1]->n_neurons; n1++)
            for (n2 = 0; n2 < net->layers[l]->n_neurons; n2++)
                net->weights[l][n1][n2] = weights[i++];
}

void network_get_weights(struct network *net, float *weights)
{
    int l, n1, n2, i;
    i = 0;
    for (l = 1; l < net->n_layers; l++)
        for (n1 = 0; n1 < net->layers[l-1]->n_neurons; n1++)
            for (n2 = 0; n2 < net->layers[l]->n_neurons; n2++)
                weights[i++] = net->weights[l][n1][n2];
}

void network_set_biases(struct network *net, float biases[
                net->n_neurons - net->layers[0]->n_neurons])
{
    int l, n, i;
    i = 0;
    for (l = 1; l < net->n_layers; l++)
        for (n = 0; n < net->layers[l]->n_neurons; n++)
            net->biases[l][n] = biases[i++];
}

void network_get_biases(struct network *net, float biases[
                net->n_neurons - net->layers[0]->n_neurons])
{
    int l, n, i;
    i = 0;
    for (l = 1; l < net->n_layers; l++)
        for (n = 0; n < net->layers[l]->n_neurons; n++)
            biases[i++] = net->biases[l][n];
}

void calc_activs_deltas(struct network *net,
                    float input[net->layers[0]->n_neurons],
                    float output[net->layers[net->n_layers-1]->n_neurons],
                    float **activs, float **deltas)
{
    int out_neurons = net->layers[net->n_layers-1]->n_neurons;
    int n1, l;
    float output_curr[out_neurons];
    float *sums;
    float *cost_derivs; /* Derivatives of the cost function with respect to the
                        * activations at the last layer */
    float *derivs;  /* Derivative of the activation function at "sums" */

    cost_derivs = malloc(out_neurons * sizeof(float));
    derivs = malloc(out_neurons * sizeof(float));
    /* Step 1: feedforward */
    feedforward(net, input, output_curr);
    /* Get sums and activations*/
    sums = malloc(out_neurons * sizeof(float));
    for (n1 = 0; n1 < out_neurons; n1++) {
        sums[n1] = net->layers[net->n_layers-1]->neurons[n1]->in_sum;
    }
    for (l = 0; l < net->n_layers; l++) {
        for (n1 = 0; n1 < net->layers[l]->n_neurons; n1++)
            activs[l][n1] =net->layers[l]->neurons[n1]->out;
    }
    /* Step 2: output error */
    /* Calculate errors in the output layer */
    vsubstract(out_neurons, cost_derivs, activs[net->n_layers-1], output);
    diff_activation_function_vector(out_neurons, derivs, sums);  
    vscalarprod(out_neurons, deltas[net->n_layers-1], cost_derivs, derivs);
    /* Step 3: backpropagate */
    for (l = net->n_layers-2; l >= 0; l--) {
        /* Compute the delta of each neuron */
        free(sums);
        sums = malloc(net->layers[l]->n_neurons * sizeof(float));
        for (n1 = 0; n1 < net->layers[l]->n_neurons; n1++) {
            /* Fill vector of sums from current layer */
            sums[n1] = net->layers[l]->neurons[n1]->in_sum;
            /* Compute weighted sum of activations from next layer */
            deltas[l][n1] = vprod(net->layers[l+1]->n_neurons,
                                  net->weights[l+1][n1], deltas[l+1]);
        }
        /* Compute derivative of the sums */
        diff_activation_function_vector(net->layers[l]->n_neurons,
        /* Compute errors of current layer (deltas) */
        vscalarprod(net->layers[l]->n_neurons, deltas[l], deltas[l], sums);
    } 
    free(sums);
    free(cost_derivs);
    free(derivs);
}

void network_backprop(struct network *net, int batch_size,
                    int out_neurons, float input[][net->layers[0]->n_neurons],
                    float output[][out_neurons], float eta, int offset,
                    float ***activs, float ***deltas)
{
    int n1, n2, l;
    int set, i;
    float gradient;
    /* Compute gradient for each test */
    for (set = 0; set < batch_size; set++)
        calc_activs_deltas(net, input[set+offset], output[set+offset],
                       activs[set], deltas[set]);
    for (l = 1; l < net->n_layers; l++) {
        for (n2 = 0; n2 < net->layers[l]->n_neurons; n2++) {
            /* Update weight */
            for (n1 = 0; n1 < net->layers[l-1]->n_neurons; n1++) {
                gradient = 0;
                for (set = 0; set < batch_size; set++)
                    gradient +=activs[set][l-1][n1]*deltas[set][l][n2];
                net->weights[l][n1][n2] -= eta/(float)batch_size * gradient;
            }
            /* Update bias */
            gradient = 0;
            for (set = 0; set < batch_size; set++)
                gradient += deltas[set][l][n2];
            net->biases[l][n2] -= eta/(float)batch_size * gradient;
        }
    }
}

void network_update_minibatch(struct network *net, int batch_size,
                    float input[][net->layers[0]->n_neurons],
                    float output[][net->layers[net->n_layers-1]->n_neurons],
                    float eta, int offset)
{
    int i, train, out_neurons = net->layers[net->n_layers-1]->n_neurons;
    float cost, cost_last, output_curr[batch_size][out_neurons];
   
    float ***deltas; /* Errors */
    float ***activs;  /* Activations */
    deltas = malloc(batch_size * sizeof(float *));
    activs = malloc(batch_size * sizeof(float *));
    for (train = 0; train < batch_size; train++) {
        deltas[train] = malloc(net->n_layers * sizeof(float *));
        activs[train] = malloc(net->n_layers * sizeof(float *));
        for (i = 0; i < net->n_layers; i++) {
            deltas[train][i] =
                    malloc(net->layers[i]->n_neurons * sizeof(float));
            activs[train][i] = 
                    malloc(net->layers[i]->n_neurons * sizeof(float));
        }
    }
    network_backprop(net, batch_size, out_neurons, input, output, eta,
                     offset, activs, deltas);
    for (train = 0; train < batch_size; train++) {
        for (i = 0; i < net->n_layers; i++) {
            free(deltas[train][i]);
            free(activs[train][i]);
        }
        free(deltas[train]);
        free(activs[train]);
    }
    free(deltas);
    free(activs);
}

/* network_SGD: train the network by stochastic gradient method.
 * For each epoch the training set is randomized and train_size/batch_size
 * mini-batches are used to perform gradient-descent
 *
 */
void network_SGD(struct network *net, int train_size, int batch_size,
     int n_epochs,
     float train_input[train_size][net->layers[0]->n_neurons],
     float train_output[train_size][net->layers[net->n_layers-1]->n_neurons],
     float eta, void fun(struct network *, int))
{
    int n_out = net->layers[net->n_layers-1]->n_neurons;
    int n_in = net->layers[0]->n_neurons;
    int batch;
    int epoch;
    int n_batches = train_size / batch_size;

    float input[batch_size][n_in], output[batch_size][n_out];
    
    for (epoch = 0; epoch < n_epochs; epoch++) {
        /* Shuffle training set */
        shuffle(train_size, n_in, train_input, n_out, train_output);
        for (batch = 0; batch < n_batches; batch++) {
            /* Create batch, and train network */
            network_update_minibatch(net, batch_size,
                                train_input, train_output, eta, batch);
        }
        if (fun) {
            fun(net, epoch);
        }
    }
}

/* shuffle: Durstenfeld's version of Fisher–Yates shuffle, for two arrays
 */
void shuffle(int len, int n_in, float inputs[len][n_in], int n_out,
             float outputs[len][n_out])
{
    int i, j;
    srand(time(NULL));
    for (i = len-1; i > 0; i--) {
        j = (int)(((float)rand() / (float)RAND_MAX) * len);
        exchange(n_in, inputs[i], inputs[j]);
        exchange(n_out, outputs[i], outputs[j]);
    }
}

/* exchange: exchange array elements */
void exchange(int n_elems, float array1[n_elems], float array2[n_elems])
{
    int i;
    float tmp;
    for (i = 0; i < n_elems; i++) {
        tmp = array1[i];
        array1[i] = array2[i];
        array2[i] = tmp;
    }
}
        



float activation_function(float x)
{
    return 1/(1 + exp(-x));
}

void diff_activation_function_vector(int n, float *out, float *in)
{
    int i;
    for (i = 0; i < n; i++) {
        out[i] = activation_function(in[i]);
        out[i] = out[i] * (1 - out[i]);
    }
}

float cost_function_batch(int n, int n_tests, float output[][n],
                          float output_correct[][n]) {
    int i;
    float cost = 0;
    for (i = 0; i < n_tests; i++)
        cost += cost_function(n, output[i], output_correct[i]);
    return cost / (float) n_tests;
}

float cost_function(int n, float output[n], float output_correct[n])
{
    float cost = 0;
    int i;
    for (i = 0; i < n; i++)
        cost += (output[i] - output_correct[i]) * (output[i]- output_correct[i]);
    return cost / (float)n;
}

int network_save_to_file(struct network *net, char *str)
{
    int l, n1, n2, fp = open(str, O_WRONLY | O_CREAT | O_TRUNC,
                             S_IRUSR | S_IWUSR);
    if (fp < 0) {
        fprintf(stderr, "Could not open file %s\n", str);
        return fp;
    }

    for (l = 1; l < net->n_layers; l++) {
        for (n2 = 0; n2 < net->layers[l]->n_neurons; n2++) {
            for (n1 = 0; n1 < net->layers[l-1]->n_neurons; n1++)
                write(fp, &(net->weights[l][n1][n2]), sizeof(float));
            write(fp, &(net->biases[l][n2]), sizeof(float));
        }
    }
    return 0;
}

int network_load_from_file(struct network *net, char *str)
{
    int l, n1, n2, fp = open(str, O_RDONLY, S_IRUSR);

    if (fp < 0) {
        fprintf(stderr, "Could not open file %s\n", str);
        return fp;
    }

    for (l = 1; l < net->n_layers; l++) {
        for (n2 = 0; n2 < net->layers[l]->n_neurons; n2++) {
            for (n1 = 0; n1 < net->layers[l-1]->n_neurons; n1++)
                read(fp, &(net->weights[l][n1][n2]), sizeof(float));
            read(fp, &(net->biases[l][n2]), sizeof(float));
        }
    }
    return 0;
}

int __new_network_save_to_file(struct network *net, char *str)
{
    int l, n1, n2, fp = open(str, O_WRONLY | O_CREAT | O_TRUNC,
                             S_IRUSR | S_IWUSR);
    if (fp < 0) {
        fprintf(stderr, "Could not open file %s\n", str);
        return fp;
    }

    for (l = 1; l < net->n_layers; l++) {
        for (n2 = 0; n2 < net->layers[l]->n_neurons; n2++) {
            for (n1 = 0; n1 < net->layers[l-1]->n_neurons; n1++)
                write(fp, &(net->weights[l][n1][n2]), sizeof(float));
            write(fp, &(net->biases[l][n2]), sizeof(float));
        }
    }
    return 0;
}

int __new_network_load_from_file(struct network *net, char *str)
{
    int l, n1, n2, fp = open(str, O_RDONLY, S_IRUSR);

    if (fp < 0) {
        fprintf(stderr, "Could not open file %s\n", str);
        return fp;
    }

    for (l = 1; l < net->n_layers; l++) {
        for (n2 = 0; n2 < net->layers[l]->n_neurons; n2++) {
            for (n1 = 0; n1 < net->layers[l-1]->n_neurons; n1++)
                read(fp, &(net->weights[l][n1][n2]), sizeof(float));
            read(fp, &(net->biases[l][n2]), sizeof(float));
        }
    }
    return 0;
}
