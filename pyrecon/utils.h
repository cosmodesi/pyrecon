#ifndef	_UTILS_H_
#define	_UTILS_H_

#include <math.h>
#include <stdio.h>

#define NDIM 3


void* my_malloc(size_t N, size_t size)
{
    void *x = malloc(N*size);
    if (x == NULL){
        fprintf(stderr, "malloc for %zu elements with %zu bytes failed...\n", N, size);
        perror(NULL);
    }
    return x;
}

void* my_calloc(size_t N, size_t size)
{
    void *x = calloc(N, size);
    if (x == NULL){
        fprintf(stderr, "calloc for %zu elements with %zu bytes failed...\n", N, size);
        perror(NULL);
    }
    return x;
}

#endif
