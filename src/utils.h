#ifndef	_UTILS_H_
#define	_UTILS_H_

#include <math.h>

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

#endif
