#ifndef	_UTILS_H_
#define	_UTILS_H_

#include <math.h>
#include <stdio.h>

#define NDIM 3

#ifdef FLOAT32
typedef float FLOAT;
#define ABS fabsf
#define SQRT sqrtf
#define POW powf
#else
typedef double FLOAT;
#define ABS fabs
#define SQRT sqrt
#define POW pow
#endif //_FLOAT32

void* my_malloc(size_t size, size_t N)
{
    void *x = malloc(N*size);
    if (x == NULL){
        fprintf(stderr, "malloc for %zu elements with %zu bytes failed...\n", N, size);
        perror(NULL);
    }
    return x;
}

void* my_calloc(size_t size, size_t N)
{
    void *x = calloc(size, N);
    if (x == NULL){
        fprintf(stderr, "calloc for %zu elements with %zu bytes failed...\n", N, size);
        perror(NULL);
    }
    return x;
}

#endif
