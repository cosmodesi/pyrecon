#define FLOAT float
#define mkname(a) a ## _ ## float
#include "_multigrid_generics.h"
#undef FLOAT
#undef mkname
#define mkname(a) a ## _ ## double
#define FLOAT double
#include "_multigrid_generics.h"
#undef FLOAT
#undef mkname


#if defined(_OPENMP)
#include <omp.h>

void set_num_threads(int num_threads)
{
  if (num_threads>0) omp_set_num_threads(num_threads);
}
#endif
