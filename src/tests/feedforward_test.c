#include <neuron.h>
#include <stdio.h>

#define OUT 3
#define HIDDEN 3
#define IN 2
int main(int argc, char *argv[])
{
  int net_structure[3] = {IN, HIDDEN, OUT};
  double input[1] = {1};
  double output[OUT];
  struct network net = create_network(3, net_structure);
  feedforward(net, input, output);
  printf("destroying\n");
  destroy_network(net);
  print_array_double(OUT, output);
  return 0;
}
