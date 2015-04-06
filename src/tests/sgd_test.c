#include <neuron.h>
#include <stdio.h>

#define EPOCHS 2
#define BATCH_SIZE 2

/*
 * data
 * 1 2 3 4
 * 1 2 3 4
 *
 * labels
 * 2 4 6 8
 */



int main(int argc, char *argv[])
{
  int structure[3] = {2, 4, 1};
  struct network net = create_network(3, structure);

  matrix_double data = alloc_matrix_double(2, 4);
  matrix_double labels = alloc_matrix_double(1, 4);

  data.data[0][0] = 1;
  data.data[1][0] = 1;
  data.data[0][1] = 2;
  data.data[1][1] = 2;
  data.data[0][2] = 3;
  data.data[1][2] = 3;
  data.data[0][3] = 4;
  data.data[1][3] = 4;

  labels.data[0][0] = 2;
  labels.data[0][1] = 4;
  labels.data[0][2] = 6;
  labels.data[0][3] = 8;

  printf("data:\n");
  print_matrix_double(data);
  printf("labels:\n");
  print_matrix_double(labels);

  network_SGD(net, data, labels, EPOCHS, BATCH_SIZE, 3);

  destroy_network(net);
  free_matrix_double(data);
  free_matrix_double(labels);
  return 0;
}
