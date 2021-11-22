CC ?= gcc
OMPFLAG ?= -fopenmp
OPT ?= $(OMPFLAG) -pedantic -Wall -Wextra

SOURCEFILES = mesh.c multigrid.c
HEADERFILES = utils.h
LIBDIR ?= ../lib

vpath %.so $(LIBDIR)

%_float.o: %.c $(HEADERFILES)
	$(CC) $(OPT) -DFLOAT32 -fPIC -O3 -c $< -o $@

%_double.o: %.c $(HEADERFILES)
	$(CC) $(OPT) -fPIC -O3 -c $< -o $@

%.so: %.o
	@mkdir -p $(LIBDIR)
	$(CC) $(OPT) -shared $< -o $(LIBDIR)/$@

all: mesh_float.so multigrid_float.so mesh_double.so multigrid_double.so

clean:
	rm -rf $(LIBDIR)

.PHONY: all clean
