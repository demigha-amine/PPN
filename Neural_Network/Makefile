CC=gcc

CFLAGS=-g -Wall


FILES=main.c  Neural_Network.c ../Matrix/matrice.c ../Matrix/oper_mat.c ../Activation/activations.c


all: exe

exe:
	$(CC) $(CFLAGS) $(FILES) -lm -o $@ 


clean:
	@rm -Rf exe

