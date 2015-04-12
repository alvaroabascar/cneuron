#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "neuron.h"
#include "matrix.h"

#undef RAND_MAX
#define RAND_MAX 59999

#define EPOCHES 10
#define BATCHES 600
#define BATCH_SIZE 10

#define TRAIN_IMG_PATH "nums/train-images-idx3-ubyte"
#define TRAIN_LABEL_PATH "nums/train-labels-idx1-ubyte"
#define TEST_IMG_PATH "nums/t10k-images-idx3-ubyte"
#define TEST_LABEL_PATH "nums/t10k-labels-idx1-ubyte"

static float training_labels[60000][10];
static float training_images[60000][784];
static float testing_labels[10000][10];
static float testing_images[10000][784];

void get_images_labels()
{
    int i, x, y;
    unsigned char dum;

    int train_img = open(TRAIN_IMG_PATH, O_RDONLY, S_IRUSR);
    int train_lab = open(TRAIN_LABEL_PATH, O_RDONLY, S_IRUSR);
    int test_img = open(TEST_IMG_PATH, O_RDONLY, S_IRUSR);
    int test_lab = open(TEST_LABEL_PATH, O_RDONLY, S_IRUSR);
    
    printf("Reading training data... ");
    lseek(train_lab, 8, SEEK_SET);
    lseek(train_img, 16, SEEK_SET);
    for (i = 0; i < 60000; i++) {
        read(train_lab, &dum, 1);
        for (x = 0; x < 10; x++) {
            training_labels[i][x] = (dum == x) ? 1 : 0;
        }
        for (y = 0; y < 784; y++) {
            read(train_img, &dum, 1);
            training_images[i][y] = ((float)dum) / ((float)255);
        }
    }
    printf("[OK]\n");

    printf("Reading test data... ");
    lseek(test_lab, 8, SEEK_SET); /* offset 8 */
    lseek(test_img, 16, SEEK_SET);
    for (i = 0; i < 10000; i++) {
        read(test_lab, &dum, 1);
        for (x = 0; x < 10; x++) {
            testing_labels[i][x] = (dum == x) ? 1 : 0;
        }
        for (y = 0; y < 784; y++) {
           read(test_img, &dum, 1);
           testing_images[i][y] = (float) dum / 255;
        }
    }
    printf("[OK]\n");
}

void test(struct network *net, int epoch) {
    int i;
    int hits = 0;
    static maxhits = 0;
    float output[net->layers[net->n_layers-1]->n_neurons];
    for (i = 0; i < 10000; i++) {
        feedforward(net, testing_images[i], output);
        if (testing_labels[i][max_index(10, output)] == 1)
            hits++;
    }
    if (hits > maxhits) {
        maxhits = hits;
        network_save_to_file(net, "mynet.net");
    }
    printf("Epoch %d: %d / 10000 (%.2f%%)\n", epoch, hits,(float)hits/100);
}

int main(int argc, char *argv[])
{
    struct network *net;
    get_images_labels();
    int net_structure[3] = {784, 30, 10};

    printf("Creating network... ");
    net = create_network(3, net_structure);
    network_load_from_file(net, "mynet.net");
    printf("[OK]\n");
    network_SGD(net, 60000, BATCH_SIZE, EPOCHES, training_images,
                training_labels, 10, test);
}
