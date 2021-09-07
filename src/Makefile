CC ?= gcc
OMPFLAG ?= -fopenmp
OPT ?= $(OMPFLAG) -pedantic -Wall -Wextra

SOURCEFILES = mesh.c multigrid.c
HEADERFILES = utils.h
LIBDIR ?= ../lib

vpath %.so $(LIBDIR)

%_float.so: %.c $(HEADERFILES)
	@mkdir -p $(LIBDIR)
	$(CC) $(OPT) -DFLOAT32 -fPIC -shared $< -o $(LIBDIR)/$@

%_double.so: %.c $(HEADERFILES)
	@mkdir -p $(LIBDIR)
	$(CC) $(OPT) -fPIC -shared $< -o $(LIBDIR)/$@

all: mesh_float.so multigrid_float.so mesh_double.so multigrid_double.so

clean:
	rm -rf $(LIBDIR)

.PHONY: all clean
