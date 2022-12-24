CC=gcc

CFLAGS=-g -Wall


FILES=main.c ../Matrix/matrice.c ../Matrix/oper_mat.c activations.c


all: exe

exe:
	$(CC) $(CFLAGS) $(FILES) -lm -o $@ 


clean:
	@rm -Rf exe

