#include "MnistRW.h"

#define TRDATASIZE 60000
#define TSDATASIZE 10000


int main() {
    uint8_t *images,*labels;

    NeuralNetwork* net = create_network(IMAGE_SIZE, 300, OUTPUT_SIZE, 0.1);
    


    // Read images & Label from MNIST TRAINING DATA

    FILE *images_file = fopen("mnist/train-images-idx3-ubyte", "rb");
    FILE *labels_file = fopen("mnist/train-labels-idx1-ubyte", "rb");

    images = (uint8_t*)malloc(TRDATASIZE * IMAGE_SIZE);
    labels = (uint8_t*)malloc(TRDATASIZE);

    Read_mnist_Imgs(images, images_file);
    Read_mnist_Labels(labels, labels_file);
    
    

    // apply_lowpass_filter(images, rows, columns, 0.2);

    // preprocess_image(images, 28, 28, 60000);

    train_batch_imgs(net,images,labels, 60000, 1);


     
    // Read images & Label from MNIST TEST DATA

    images_file = fopen("mnist/t10k-images-idx3-ubyte", "rb");
    labels_file = fopen("mnist/t10k-labels-idx1-ubyte", "rb");


    images = (uint8_t*)malloc(TSDATASIZE * IMAGE_SIZE);
    labels = (uint8_t*)malloc(TSDATASIZE);

    Read_mnist_Imgs(images, images_file);
    Read_mnist_Labels(labels, labels_file);


    // preprocess_image(images, 28, 28, 10000);


    float NET_RATE = predict_rate_network(net, images, labels, 10000, 1);


    printf("%1.6f\n", 
	 NET_RATE);

return 0;
}