#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <neuron.h>
#include <matrix.h>
#include <random.h>

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

/* Add random weights and biases, distributed normally with mean 0 and
 * standard deviation 1.
 */
void set_random_weights_biases(struct network net)
{
  int i, j, k;
  long seed = time(NULL);
  for (i = 0; i < net.n_layers-1; i++) {
    for (j = 0; j < net.net_structure[i+1]; j++) {
      net.biases[i].data[j][0] = (double) gauss0(&seed);
      for (k = 0; k < net.net_structure[i]; k++) {
        net.weights[i].data[j][k] = (double) gauss0(&seed);
      }
    }
  }
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

/* Save weights and biases in binary format, to the specified file */
void save_network(struct network net, char *filename)
{
  int l, i, j, n_neurons, n_neurons_prev;
  FILE *ptr = fopen(filename, "wb");
  /* an int indicating the number of layers */
  fwrite(&net.n_layers, sizeof(net.n_layers), 1, ptr);
  /* an int per layer indicating the number of neurons in that layer */
  fwrite(net.net_structure, sizeof(net.net_structure[l]), net.n_layers, ptr);
  /* for each layer, all the biases and all the weights */
  for (l = 0; l < net.n_layers-1; l++) {
    n_neurons = net.net_structure[l+1];
    n_neurons_prev = net.net_structure[l];
    /* save all biases and weights of the layer */
    for (i = 0; i < n_neurons; i++) {
      fwrite(&net.biases[l].data[i][0], sizeof(double), 1, ptr);
      for (j = 0; j < n_neurons_prev; j++) {
        fwrite(&net.weights[l].data[i][j], sizeof(double), 1, ptr);
      }
    }
  }
  fclose(ptr);
}

/* Load weights and biases in binary format from the specified file */
struct network load_network(char *filename)
{
  int l, i, j, n_layers, *net_structure, n_neurons, n_neurons_prev;
  FILE *ptr = fopen(filename, "rb");
  /* read number of layers */
  fread(&n_layers, sizeof(int), 1, ptr);
  /* allocate space for array of ints (one int per layer) */
  net_structure = malloc(n_layers * sizeof(int));
  fread(net_structure, sizeof(int), n_layers, ptr);
  struct network net = create_network(n_layers, net_structure);
  for (l = 0; l < n_layers-1; l++) {
    n_neurons = net_structure[l+1];
    n_neurons_prev = net_structure[l];
    for (i = 0; i < n_neurons; i++) {
      fread(&net.biases[l].data[i][0], sizeof(double), 1, ptr);
      for (j = 0; j < n_neurons_prev; j++) {
        fread(&net.weights[l].data[i][j], sizeof(double), 1, ptr);
      }
    }
  }
  free(net_structure);
  return net;
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
