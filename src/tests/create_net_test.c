#include <neuron.h>

int main(int argc, char *argv[])
{
  int net_structure[3] = {10, 20, 2};
  struct network net = create_network(3, net_structure);
  destroy_network(net);
  return 0;
}
