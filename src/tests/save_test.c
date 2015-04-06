#include <neuron.h>
#include <stdio.h>

#define IN 2
#define HIDDEN 2
#define OUT 1

int main(int argc, char *argv[])
{
  int net_structure[3] = {IN, HIDDEN, OUT};
  double input[IN] = {1, 2};
  double output[OUT], output2[OUT];

  struct network net2, net = create_network(3, net_structure);
  set_random_weights_biases(net);
  feedforward(net, input, output);

  save_network(net, "net.net");
  destroy_network(net);

  print_array_double(OUT, output);

  net2 = load_network("net.net");
  feedforward(net2, input, output2);
  print_array_double(OUT, output2);
  destroy_network(net2);

  return 0;
}
