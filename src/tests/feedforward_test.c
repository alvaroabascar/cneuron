#include <neuron.h>
#include <stdio.h>

#define IN 2
#define HIDDEN 2
#define OUT 1
int main(int argc, char *argv[])
{
  int net_structure[3] = {IN, HIDDEN, OUT};
  double input[IN] = {1, 2};
  double output[OUT];
  struct network net = load_network("feedforward_test.net");
  feedforward(net, input, output);
  destroy_network(net);
  printf("correct output: 0.6290\nactual output:\n");
  print_array_double(OUT, output);
  return 0;
}
