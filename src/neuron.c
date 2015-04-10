#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <neuron.h>
#include <matrix.h>
#include <random.h>

/**
 * Create a neural network with as many layers as indicated with the int
 * "n_layers". Each layer will contain the number of neurons indicated by
 * the corresponding position of the array "net_structure". Eg. layer 3
 * will have net_structure[3] neurons.
 *
 * The network is dynamically allocated with mallocs, and thus must be
 * manually destroyed after done working with it, calling destroy_network.
 */
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

/**
 * Add random weights and biases, distributed normally with mean 0 and
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

/**
 * Release the space allocated for a neural network.
 */
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

/**
 * given an input to the network (an array of activations of the first
 * layer), perform a feedforward pass and return the output (activations
 * of the last layer) in the array "output", which must have enough space
 * allocated prior the function call.
 */
void feedforward(struct network net, double *input, double *output)
{
  int l;
  matrix_double activation, zs;
  /* fill activation (vertical vector) with the inputs */
  activation = alloc_matrix_double(net.net_structure[0], 1);
  set_col_matrix_double(activation, input, 0);

  for (l = 1; l < net.n_layers; l++) {
    /* allocate a matrix (1 column) for the weighted inputs of layer "l" */
    zs = matrix_product_matrix_double(net.weights[l-1], activation);
    free_matrix_double(activation);
    /* we must apply the sigmoid function to get the activations. */
    activation = copy_matrix_double(zs);
    vectorized_sigma(activation);
    free_matrix_double(zs);
  }
  /* copy the last activations (output from the net) into the output vector */
  copy_col_matrix_double(activation, output, 0);
  free_matrix_double(activation);
}

/**
 * Save weights and biases in binary format, to the specified file.
 */
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

/**
 * Load weights and biases in binary format from the specified file.
 */
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
  fclose(ptr);
  return net;
}

/**
 * Stochastic Gradient Descent
 *
 * Parameters:
 *   net -> network to be trained.
 *   training_data -> matrix of inputs. Each column is an input.
 *   training_data_labels -> matrix of outputs. Each column is an output.
 *   epochs -> number of runs across the whole training set.
 *   mini_batch_size -> number of training inputs to use in each
 *                      run of the backpropagation.
 *
 */
void network_SGD(struct network net, matrix_double training_data,
                 matrix_double training_labels, int epochs,
                 int mini_batch_size, double eta)
{
  int i, j, k, data_size = training_data.ncols;
  matrix_double mini_batch_data, mini_batch_labels;
  struct pair_coordinates section;
  for (i = 0; i < epochs; i++) {
    shuffle_data(training_data, training_labels);
    for (j = 0; j < data_size; j += mini_batch_size) {
      /* extract a mini batch of size mini_batch_size (if possible)
       * or with all the remaining training cases.
       */
      k = j + mini_batch_size;
      k = k > data_size ? data_size : k;
      section.a = (struct coordinate) { .row = 0, .col = j };
      section.b = (struct coordinate) { .row = training_data.nrows, .col = k };
      mini_batch_data = extract_section_matrix_double(training_data, section);
      section.b.row = training_labels.nrows;
      mini_batch_labels = extract_section_matrix_double(training_labels,
                                                        section);
      update_minibatch(net, mini_batch_data, mini_batch_labels);
      free_matrix_double(mini_batch_data);
      free_matrix_double(mini_batch_labels);
    }
  }
}

/**
 * Update network weights and biases applying gradient descent with
 * backpropagation to a single minibatch. The minibatch is composed by
 * a matrix of inputs (each column is an input) and a matrix of ordered
 * labels (correct outputs, each column is an output).
 *
 *
 * Inputs:
 *    net -> the network
 *    training_data -> a matrix_double which contains a set of training
 *                     inputs (one input per column).
 *    training_labels -> a matrix_double which contains the corresponding
 *                       outputs (labels), one per column. The ith label
 *                       is the correct output of the ith training input.
 */
void update_minibatch(struct network net, matrix_double training_data, matrix_double training_labels)
{
  /* We must compute the gradient with respect to the biases and weights of the
   * network. The first step is to compute the errors in the output layer. In
   * order to do so, we must do a feedforward pass for every input. Along the
   * way we will store all the weighted inputs (zs) and activations (activs).
   *
   * zs[l][i] is matrix (array actually, cause it's got one single column) of
   * weighted inputs of layer "l", when the input is the number "i".
   *
   * The same holds for activs[l][i].
   */
  int i, l;
  double input[training_data.nrows];
  double output[training_labels.nrows];
  /* gradient from a single input */
  matrix_double nabla_b, nabla_w;
  /* sum of gradients (from all inputs) */
  matrix_double sum_nabla_b, sum_nabla_w;
  /* for each training input... */
  for (i = 0; i < training_data.ncols; i++) {
    copy_col_matrix_double(training_data, input, i);
    copy_col_matrix_double(training_labels, output, i);
    network_backprop(net, input, output, nabla_w, nabla_b);
  }

  /* Cleanup */
  free_matrix_double(nabla_b);
  free_matrix_double(nabla_w);
}

/**
 * Backpropagation using gradient descent.
 *
 * Input:
 *
 *   net -> the neural network to be trained.
 *
 *   input -> array of doubles containing an input to the network.
 *
 *   output -> array of doubles containing the expected output.
 *
 *   nabla_w -> matrix_double which will be filled with the gradients
 *              with respect to the weights of the network.
 *
 *   nabla_b -> matrix_double which will be filled with the gradients
 *              with respect to the weights of the network.
 *
 */
void network_backprop(struct network net, double *input, double *output,
                      matrix_double nabla_w, matrix_double nabla_b)
{
  int l, n_layers = net.n_layers;
  matrix_double zs[n_layers];
  matrix_double activs[n_layers];
  matrix_double costs;
  /* this matrix is just used to calculate the costs later */
  matrix_double output_tmp = alloc_matrix_double(
                                 net.net_structure[n_layers], 1);
  set_col_matrix_double(output_tmp, output, 0);

  /* Step 1: set the inputs to the network */
  activs[0] = alloc_matrix_double(net.net_structure[0], 1);
  set_col_matrix_double(activs[0], input, 0);
  /* zs[0] is unused, but it's easier to allocate it, for the cleanup */
  zs[0] = alloc_matrix_double(0, 0);

  /* Step 2: do the feedforward pass */
  for (l = 1; l < n_layers; l++) {
    zs[l] = matrix_product_matrix_double(net.weights[l-1], activs[l-1]);
    activs[l] = copy_matrix_double(zs[l]);
    vectorized_sigma(activs[l]);
  }
  costs = cost_derivatives(activs[n_layers], output_tmp);

  /* Cleanup */
  free_matrix_double(costs);
  for (l = 0; l < n_layers; l++) {
    free_matrix_double(activs[l]);
    free_matrix_double(zs[l]);
  }
}

/**
 * Return a vector (matrix with a single row) of partial derivatives of the
 * cost. The quadratic cost is:
 *            C = 0.5 * sum_over_i(output_i - correct_output_i)^2
 *
 * And thus the derivative with respect to i is just:
 *            dC/di = output_i - correct_output_i
 *
 * Input:
 *        output -> matrix of output from the network (single column)
 *        correct_output -> matrix of correct outputs from the network (single
 *                          column)
 *
 * Output:
 *        a matrix_double with a single row. Column "i" corresponds to the
 *        partial derivative of the cost with respect to output from neuron
 *        "i".
 */
matrix_double cost_derivatives(matrix_double output, matrix_double correct_output)
{
  matrix_double result = copy_matrix_double(output);
  /* substract from result (which contains now the actual outputs) the
   * correct outputs */
  substract_matrix_from_matrix_double(correct_output, result);
  return result;
}

/**
 * Given a couple of matrix_double, one with the training inputs (data) and
 * another with the labels (labels), shuffles the columns of both matices,
 * but conserving the correspondence of columns from "data" and "labels".
 *
 * The algorithm used is Fisher-Yates.
 */
void shuffle_data(matrix_double data, matrix_double labels)
{
  int i, j;
  for (i = data.ncols-1; i >= 1; i--) {
    j = rand_lim(i);
    interchange_cols_matrix_double(data, i, j);
    interchange_cols_matrix_double(labels, i, j);
  }
}

/**
 * Apply the sigmoid function to all elements of matrix. The original matrix
 * is altered.
 */
void vectorized_sigma(matrix_double matrix)
{
  int i, j;
  for (i = 0; i < matrix.nrows; i++)
    for (j = 0; j < matrix.ncols; j++)
      matrix.data[i][j] = sigma(matrix.data[i][j]);
}

/**
 * Sigmoid function: s(x) = 1 / (1 + exp(-x))
 */
double sigma(double x)
{
  return 1 / (1 + exp(-x));
}
