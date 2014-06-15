#include <stdio.h>
#include "neuron.h"

int main()
{
    int structure[3] = {2, 2, 1};
    float input[2] = {1,2};
    float output[1];
    struct network *net = create_network(3, structure);

    feedforward(net, input, output);
    vprint_float(1, output);
    network_save_to_file(net, "mynet.net");
    
    srand(time(NULL));
    network_set_random_weights_biases(net, -1, 1);

    network_load_from_file(net, "mynet.net");
    feedforward(net, input, output);
    vprint_float(1, output);
}
