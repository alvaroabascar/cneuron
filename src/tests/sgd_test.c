#include <neuron.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
  int structure[3] = {2, 4, 1};
  struct network net = create_network(3, structure);

  destroy_network(net);
  return 0;
}
