CC=gcc

CFLAGS=-g -Wall


FILES=main.c  Neural_Network.c ../Matrix/matrice.c ../Matrix/oper_mat.c ../Activation/activations.c ../mnist_reader/mnist_reader.c



all: exe

exe:
	$(CC) $(CFLAGS) $(FILES) -lm -o $@ 


clean:
	@rm -Rf exe

