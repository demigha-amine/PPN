#include "../Neural_Network/Neural_Network.h"
#include <stdio.h>
#include <time.h>

#define MAX_SIZE 8000
#define LEARNING_RATE 0.1
#define num_folds 5

double cross_validate(int size, int HIDDEN_NODES, int test_offset, uint8_t *images, uint8_t *labels,uint8_t *im, uint8_t *lb) {
    int fold_size = size / num_folds;
    double sum_accuracy = 0;


    for (int fold = 0; fold < num_folds; ++fold) {
        // Créez un nouveau réseau de neurones pour chaque itération
        NeuralNetwork* net = create_network(IMAGE_SIZE, HIDDEN_NODES, OUTPUT_SIZE, LEARNING_RATE);

        clock_t trainin_begin = clock();
        // Entraînez le réseau sur tous les plis/folds sauf le pli actuel
        for (int i = 0; i < num_folds; ++i) {
            images = im;
            labels = lb;

            if (i == fold) {
                continue;
            }

            int start = i * fold_size;
            // Problème du training sur le 2ème fold
            train_batch_imgs(net, images + start * IMAGE_SIZE, labels + start, fold_size);

        }
        clock_t trainin_end = clock();

        // Traning time pour chaque fold
        double training_delta = (double) (trainin_end - trainin_begin) / CLOCKS_PER_SEC;
        printf("training delta = %lf\n",training_delta);
        // Testez le réseau sur le pli actuel
        int start = fold * fold_size;
        double accuracy = predict_rate_network(net, images + start * IMAGE_SIZE, labels + start, fold_size);

        sum_accuracy += accuracy;

        // Libérez le réseau de neurones pour cette itération
 
        free_network(net);

    }

    // Calculez et retournez la moyenne des précisions
    return sum_accuracy / num_folds;
}

int main(int argc, char **argv) {

    if (argc != 4) {
        fprintf(stderr, "Error! Expecting :    ./exe	Training size	Hidden Nodes	Test Offset\n");
        return 1;
    }
    int size = atoi(argv[1]);
    int HIDDEN_NODES = atoi(argv[2]);
    int test_offset = atoi(argv[3]);

    FILE* imageFile = fopen("./mnist_reader/mnist/train-images-idx3-ubyte", "r");
    FILE* labelFile = fopen("./mnist_reader/mnist/train-labels-idx1-ubyte", "r");

    // Read size images from the MAX_SIZE images
    uint8_t* images = readMnistImages(imageFile, MAX_SIZE, size);
    uint8_t* labels = readMnistLabels(labelFile, MAX_SIZE, size);

    FILE* imag = fopen("./mnist_reader/mnist/train-images-idx3-ubyte", "r");
    FILE* label = fopen("./mnist_reader/mnist/train-labels-idx1-ubyte", "r");

    uint8_t* im = readMnistImages(imag, MAX_SIZE, size);
    uint8_t* lb = readMnistLabels(label, MAX_SIZE, size);


    fclose(imageFile);
    fclose(labelFile);

   double cross_val_result = cross_validate(size, HIDDEN_NODES, test_offset, images, labels,im,lb);
    printf("Cross Validation Result: %f\n", cross_val_result);
  return 0;
}

