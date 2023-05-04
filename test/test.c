
#include "../Neural_Network/Neural_Network_2.h"
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <assert.h>
#include <cmocka.h>

static void test_net_rate(void **state) {

    int training_size = 10000;
	int HIDDEN_NODES = 300;
	int test_offset = 0;
	int test_size = 1000;

	//CREATE NETWORK

    NeuralNetwork* net = create_network(784, HIDDEN_NODES, 10, 0.1);


	// LOAD THE TRAINED MODEL

	// Reseau avec 1 Hidden Layer
	// net->hidden_weights = charger_mat("../Neural_Network/1Hidden/hidden_w");
	// net->hidden_bias = charger_mat("../Neural_Network/1Hidden/hidden_b");
	// net->output_weights = charger_mat("../Neural_Network/1Hidden/output_w");
	// net->output_bias = charger_mat("../Neural_Network/1Hidden/output_b");
	

	// Reseau avec 2 Hidden Layers
    net->hidden_weights = charger_mat("../Neural_Network/TwoHiDDEN/2hidden_w");
	net->hidden_weights_2 = charger_mat("../Neural_Network/TwoHiDDEN/2hidden_w_2");
	net->hidden_bias = charger_mat("../Neural_Network/TwoHiDDEN/2hidden_b");
	net->hidden_bias_2 = charger_mat("../Neural_Network/TwoHiDDEN/2hidden_b_2");
	net->output_weights = charger_mat("../Neural_Network/TwoHiDDEN/2output_w");
	net->output_bias = charger_mat("../Neural_Network/TwoHiDDEN/2output_b");
  	

	// 2 TESTING

	FILE* imageFile = fopen("../mnist_reader/mnist/t10k-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("../mnist_reader/mnist/t10k-labels-idx1-ubyte", "r");

	// Read size images from the MAX_SIZE images
	uint8_t* images = readMnistImages(imageFile, test_offset, test_size);
	uint8_t* labels = readMnistLabels(labelFile, test_offset, test_size);


	fclose(imageFile);
	fclose(labelFile);


    
	double NET_RATE = predict_rate_network(net, images, labels, test_size,1);
    

	
	// TRAINING DATASET & HIDDEN NODES PERFORMANCE

	printf("%d; %d; %d; %1.6f\n",
	 training_size,
	 HIDDEN_NODES,
	 test_offset,
	 NET_RATE);


	free_network(net);
    free(images);
    free(labels);

    assert_in_range(NET_RATE, 0.0, 100.0);

}

int main(void) {

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_net_rate),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);

    
}