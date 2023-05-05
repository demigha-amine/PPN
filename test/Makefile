CC=gcc -fopenmp

CFLAGS=-g -Wall




FILES=test.c ../Neural_Network/Neural_Network_2.c ../Matrix/matrice.c ../Matrix/oper_mat.c ../Activation/activations.c ../mnist_reader/mnist_reader.c

all: exe


exe:
	$(CC) $(CFLAGS) $(FILES) -lm -lcblas -lcmocka -o $@
	./exe

clean:
	@rm -Rf exe
