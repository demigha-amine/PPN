#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "mnist_reader.h"

int main(void) {
	FILE* imageFile = fopen("mnist/train-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("mnist/train-labels-idx1-ubyte", "r");

	// Read 10 images from the 50 image
	uint8_t* images = readMnistImages(imageFile, 8000, 3);
	uint8_t* labels = readMnistLabels(labelFile, 8000, 3);

	fclose(imageFile);
	fclose(labelFile);

	for(int i=0; i<3; i++) {
		printf("i=%d\n", i);
		printAsciiDigit(images + i*784*sizeof(uint8_t));
		printf("Label=%d\n", *(labels+i));
		printf("********************************\n");
	}

	free(images);
	free(labels);

	return 0;
}
