#include "MnistRW.h"


int reverse_int(int i) {
    uint8_t c1, c2, c3, c4;
    c1 = i & 255;                    // Masque pour récupérer le premier octet
    c2 = (i >> 8) & 255;             // Décalage de 8 bits pour récupérer le deuxième octet
    c3 = (i >> 16) & 255;            // Décalage de 16 bits pour récupérer le troisième octet
    c4 = (i >> 24) & 255;            // Décalage de 24 bits pour récupérer le quatrième octet
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;  // Concaténation des octets dans le bon ordre
}




void preprocess_image(uint8_t* image, int rows, int columns, int number_of_images){
    //Calcul de la moyenne et de l'écart type des pixels de l'image

    for (int k = 0; k < number_of_images; k++)
    {
        float sum = 0.0, sum_sq = 0.0;
        for(int i=0; i< rows * columns; i++){
            sum += image[i];
            sum_sq += image[i] * image[i];
        }

        float mean = sum / (rows * columns);
        float std_dev =sqrt(sum_sq / (rows * columns) - mean * mean);

        //Centrage et mise à l'échelle des pixels de l'image

        for(int i=0; i<rows * columns; i++){
        image[i] = (unsigned char)((image[i] - mean) / std_dev * 128 + 128);
        }
    }
    
   
    
    
}


//Fonction qui applique le filtre passe-bas sur une image
void apply_lowpass_filter(uint8_t *image, int rows, int columns,float cutoff_frequency) {
    // Calcul du rayon du filtre passe-bas
    int radius = (int)(0.5 * rows * cutoff_frequency);
    
    // Allocation d'un tampon temporaire pour l'image filtrée
    unsigned char *temp_image = (unsigned char*)malloc(rows * columns* sizeof(unsigned char));
    
    // Parcours de chaque pixel de l'image
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            float sum = 0;
            float count = 0;
            
            // Parcours de chaque pixel voisin dans le rayon du filtre
            for (int k = i - radius; k <= i + radius; k++) {
                for (int l = j - radius; l <= j + radius; l++) {
                    if (k >= 0 && k < rows && l >= 0 && l < columns) {
                        sum += (float)image[k * columns + l];
                        count += 1;
                    }
                }
            }
            
            // Calcul de la moyenne des pixels voisins
            float mean = sum / count;
            
            // Stockage du pixel filtré dans le tampon temporaire
            temp_image[i * columns + j] = (unsigned char)mean;
        }
    }
    
    // Copie de l'image filtrée dans l'image d'origine
    for (int i = 0; i < rows * columns; i++) {
        image[i] = temp_image[i];
    }
    
    // Libération du tampon temporaire
    free(temp_image);
}



void Read_mnist_Imgs(uint8_t* images, FILE* images_file) {

    // Lecture des informations d'en-tête dans les fichiers
    int magic_number_images, number_of_images, rows, columns;
  
    fread(&magic_number_images, sizeof(magic_number_images), 1, images_file);  // Lecture du magic number des images
    magic_number_images = reverse_int(magic_number_images);                    // Inversion de l'ordre des octets
    fread(&number_of_images, sizeof(number_of_images), 1, images_file);        // Lecture du nombre d'images
    number_of_images = reverse_int(number_of_images);                          // Inversion de l'ordre des octets
    fread(&rows, sizeof(rows), 1, images_file);                                 // Lecture du nombre de lignes par image
    rows = reverse_int(rows);                                                   // Inversion de l'ordre des octets
    fread(&columns, sizeof(columns), 1, images_file);                           // Lecture du nombre de colonnes par image
    columns = reverse_int(columns);                                             // Inversion de l'ordre des octets
    

    // Lecture des données d'images et d'étiquettes dans les fichiers
    fread(images, sizeof(uint8_t), number_of_images * rows * columns, images_file);
    fclose(images_file);

}


void Read_mnist_Labels(uint8_t* labels, FILE* labels_file) {
    

    // Lecture des informations d'en-tête dans les fichiers
    int  magic_number_labels, number_of_labels;
  
   

    fread(&magic_number_labels, sizeof(magic_number_labels), 1, labels_file);   // Lecture du magic number des étiquettes
    magic_number_labels = reverse_int(magic_number_labels);                     // Inversion de l'ordre des octets
    fread(&number_of_labels, sizeof(number_of_labels), 1, labels_file);         // Lecture du nombre d'étiquettes
    number_of_labels = reverse_int(number_of_labels);                           // Inversion de l'ordre des octets
    

    // Lecture des données d'images et d'étiquettes dans les fichiers
    fread(labels, sizeof(uint8_t), number_of_labels, labels_file);
    fclose(labels_file);


}

