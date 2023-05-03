CC=gcc

CFLAGS=-g -Wall


FILES=main.c  Neural_Network.c ../Matrix/matrice.c ../Matrix/oper_mat.c ../Activation/activations.c ../mnist_reader/mnist_reader.c

#FILES2=main.c  Neural_Network_2.c ./Matrix/matrice.c ./Matrix/oper_mat.c ./Activation/activations.c ./mnist_reader/mnist_reader.c




all: exe #2Hidden

exe:
	$(CC) $(CFLAGS) $(FILES) -lm -o $@

#2Hidden:
#	$(CC) $(CFLAGS) $(FILES2) -lm -o $@

clean:
	@rm -Rf exe 

