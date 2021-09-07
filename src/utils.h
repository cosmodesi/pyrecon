#ifndef	_UTILS_H_
#define	_UTILS_H_

#include <math.h>

#define NDIM 3

#ifdef FLOAT32
typedef float FLOAT;
#define ABS fabsf
#define SQRT sqrtf
#else
typedef double FLOAT;
#define ABS fabs
#define SQRT sqrt
#endif //_FLOAT32

#endif
