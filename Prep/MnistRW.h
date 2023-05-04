#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include "../Neural_Network/Neural_Network_2.h"


// Fonction pour inverser l'ordre des octets d'un entier
int reverse_int(int i);

// Fonction qui centre et mise en échelle d'une image
void preprocess_image(uint8_t* image, int rows, int columns, int number_of_images);

//Fonction qui applique le filtre passe-bas sur une image
void apply_lowpass_filter(uint8_t *image, int rows, int columns, double cutoff_frequency);

// Read Images from Mnist Data
void Read_mnist_Imgs(uint8_t* images, FILE* images_file);

// Read Labels from Mnist Data
void Read_mnist_Labels(uint8_t* labels, FILE* labels_file);


