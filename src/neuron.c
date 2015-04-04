#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <neuron.h>
#include <matrix.h>

/* allocate space and return a network */
struct network create_network(int n_layers, int *net_structure)
{
  int i, j;

  struct network net;
  net.n_layers = n_layers;

  /* biases: one array per layer */
  net.biases = malloc((n_layers-1) * sizeof(matrix_double));

  /* weights: one matrix per layer (except the first one) */
  net.weights = malloc((n_layers-1) * sizeof(matrix_double));

  /* array indicating the number of neurons in each layer */
  net.net_structure = malloc(n_layers * sizeof(int));
  copy_array_int(n_layers, net_structure, net.net_structure);

  for (i = 0; i < n_layers-1; i++) {
    /* allocate space for all the biases of layer i */
    net.biases[i] = alloc_matrix_double(net_structure[i+1], 1);
    /* allocates space for all the weights of layer i+1
     * each neuron of layer i+1 has associated an array of weights, one
     * weight per neuron in the previous layer (i)
     */
    net.weights[i] = alloc_matrix_double(net_structure[i+1],
                                         net_structure[i]);
  }
  return net;
}

/* free the memory used for a network. */
void destroy_network(struct network net)
{
  int i, j, k;
  /* free weights and biases*/
  for (i = 0; i < net.n_layers-1; i++) {
    free_matrix_double(net.weights[i]);
    free_matrix_double(net.biases[i]);
  }
  free(net.biases);
  free(net.weights);
  /* free structure array */
  free(net.net_structure);
}

/* given an input to the network (an array of activations of the first layer),
 * perform a feedforward pass and return the output in the array "output"
 */
void feedforward(struct network net, double *input, double *output)
{
  int i, maxsize;
  /* allocate enough space for the activations and weighted inputs (zs) of
   * the largest layer
   */
  maxsize = absmax_array_int(net.n_layers, net.net_structure);
  matrix_double activations = alloc_matrix_double(net.net_structure[0], 1);
  matrix_double zs = alloc_matrix_double(net.net_structure[1], 1);
  /* first set of activations are the input to the network */
  set_column_matrix_double(activations, input, 0);

  for (i = 0; i < net.n_layers-1; i++) {
    /* multiply weights by activations, get weighted inputs of the new layer */
    zs = matrix_product_matrix_double(net.weights[i], activations);
    /* add biases */
    add_matrix_to_matrix_double(net.biases[i], zs);
    /* turn weighted input into activations */
    vectorized_sigma(zs);
    free_matrix_double(activations);
    activations = copy_matrix_double(zs);
  }
  copy_col_matrix_double(activations, output, 0);
}

/* apply sigmoid function to all elements of matrix */
void vectorized_sigma(matrix_double matrix)
{
  int i, j;
  for (i = 0; i < matrix.nrows; i++)
    for (j = 0; j < matrix.ncols; j++)
      matrix.data[i][j] = sigma(matrix.data[i][j]);
}

/* sigmoid function */
double sigma(double x)
{
  return 1 / (1 + exp(-x));
}
