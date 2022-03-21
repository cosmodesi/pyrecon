CC ?= gcc
OMPFLAG ?= -fopenmp
OPT ?= $(OMPFLAG) -pedantic -Wall -Wextra -O3
LIBDIR ?= ../lib

SOURCEFILES = mesh.c multigrid.c
HEADERFILES = utils.h

vpath %.so $(LIBDIR)

%_float.o: %.c $(HEADERFILES)
	$(CC) $(OPT) -DFLOAT32 -fPIC -c $< -o $@

%_double.o: %.c $(HEADERFILES)
	$(CC) $(OPT) -fPIC -c $< -o $@

%.so: %.o
	@mkdir -p $(LIBDIR)
	$(CC) $(OPT) -shared $< -o $(LIBDIR)/$@

all: mesh_float.so multigrid_float.so mesh_double.so multigrid_double.so

clean:
	rm -rf $(LIBDIR)

.PHONY: all clean
