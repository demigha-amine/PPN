CC=gcc -fopenmp

CFLAGS=-g -Wall




FILES3=main_prep.c MnistRW.c  ../Neural_Network/Neural_Network_2.c ../Matrix/matrice.c ../Matrix/oper_mat.c ../Activation/activations.c ../mnist_reader/mnist_reader.c

all: prep


prep:
	$(CC) $(CFLAGS) $(FILES3) -lm -lcblas -o $@

clean:
	@rm -Rf prep

