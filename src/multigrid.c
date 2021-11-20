#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//#include "multigrid.h"
#include "utils.h"

// This is a readaptation of Martin J. White's code, available at https://github.com/martinjameswhite/recon_code
// C++ dependencies have been removed, solver parameters e.g. niterations exposed
// Grid can be non-cubic, with a cell size different along each direction (but why would we want that?)
// los can be global (to test the algorithm in the plane-parallel limit)

// The multigrid code for solving our modified Poisson-like equation.
// See the notes for details.
// The only place that the equations appear explicitly is in
// gauss_seidel, jacobi and residual
//
//
// The finite difference equation we are solving is written schematically
// as A.v=f where A is the matrix operator formed from the discretized
// partial derivatives and f is the source term (density in our case).
// The matrix, A, is never explicitly constructed.
// The solution, phi, is in v.
//
//
// Author:	Martin White	(UCB/LBNL)
// Written:	20-Apr-2015
// Modified:	20-Apr-2015
//

/*
int get_num_threads()
{
  //Calculate number of threads
  int num_threads=0;
#pragma omp parallel
  {
#pragma omp atomic
    num_threads++;
  }
  return num_threads;
}
*/

void jacobi(FLOAT *v, const FLOAT *f, const int* nmesh, const FLOAT* boxsize, const FLOAT* boxcenter, const FLOAT beta, const FLOAT damping_factor, const int niterations, const FLOAT* los) {
  // Does an update using damped Jacobi. This, and in residual below,
  // is where the explicit equation we are solving appears.
  // See notes for more details.
  const size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  const int nmeshz = nmesh[2];
  const int nmeshyz = nmesh[2]*nmesh[1];
  FLOAT* jac = (FLOAT *) malloc(size*sizeof(FLOAT));
  FLOAT cell, cell2[NDIM], icell2[NDIM], offset[NDIM], losn[NDIM];
  for (int idim=0; idim<NDIM; idim++) {
    cell = boxsize[idim]/nmesh[idim];
    cell2[idim] = cell*cell;
    icell2[idim] = 1./cell2[idim];
    offset[idim] = (boxcenter[idim] - boxsize[idim]/2.)/cell;
    if (los != NULL) losn[idim] = los[idim]/cell;
  }
  for (int iter=0; iter<niterations; iter++) {
    #pragma omp parallel for shared(v,f,jac)
    for (int ix=0; ix<nmesh[0]; ix++) {
      FLOAT px = (los == NULL) ? ix + offset[0] : losn[0];
      size_t ix0 = nmeshyz*ix;
      size_t ixp = nmeshyz*((ix+1) % nmesh[0]);
      size_t ixm = nmeshyz*((ix-1+nmesh[0]) % nmesh[0]);
      for (int iy=0; iy<nmesh[1]; iy++) {
        FLOAT py = (los == NULL) ? iy + offset[1] : losn[1];
        size_t iy0 = nmeshz*iy;
        size_t iyp = nmeshz*((iy+1) % nmesh[1]);
        size_t iym = nmeshz*((iy-1+nmesh[1]) % nmesh[1]);
        for (int iz0=0; iz0<nmesh[2]; iz0++) {
          FLOAT pz = (los == NULL) ? iz0 + offset[2] : losn[2];
          FLOAT g = beta/(cell2[0]*px*px+cell2[1]*py*py+cell2[2]*pz*pz);
          FLOAT gpx2 = icell2[0] + g*px*px;
          FLOAT gpy2 = icell2[1] + g*py*py;
          FLOAT gpz2 = icell2[2] + g*pz*pz;
          size_t izp = (iz0+1) % nmesh[2];
          size_t izm = (iz0-1+nmesh[2]) % nmesh[2];
          size_t ii = ix0 + iy0 + iz0;
          jac[ii] = f[ii]+
                    gpx2*(v[ixp+iy0+iz0]+v[ixm+iy0+iz0])+
                    gpy2*(v[ix0+iyp+iz0]+v[ix0+iym+iz0])+
                    gpz2*(v[ix0+iy0+izp]+v[ix0+iy0+izm])+
                    g/2*(px*py*(v[ixp+iyp+iz0]+v[ixm+iym+iz0]
                               -v[ixm+iyp+iz0]-v[ixp+iym+iz0])+
                    px*pz*(v[ixp+iy0+izp]+v[ixm+iy0+izm]
                          -v[ixm+iy0+izp]-v[ixp+iy0+izm])+
                    py*pz*(v[ix0+iyp+izp]+v[ix0+iym+izm]
                          -v[ix0+iym+izp]-v[ix0+iyp+izm]));
          if (los == NULL) {
            jac[ii] += g*(px*(v[ixp+iy0+iz0]-v[ixm+iy0+iz0])+
                          py*(v[ix0+iyp+iz0]-v[ix0+iym+iz0])+
                          pz*(v[ix0+iy0+izp]-v[ix0+iy0+izm]));
          }
          jac[ii] /= 2*(gpx2 + gpy2 + gpz2);
        }
      }
    }
    #pragma omp parallel for shared(v,jac)
    for (size_t ii=0; ii<size; ii++) v[ii] = (1-damping_factor)*v[ii] + damping_factor*jac[ii];
  }
  free(jac);
}


void residual(const FLOAT* v, const FLOAT* f, FLOAT* r, const int* nmesh, const FLOAT* boxsize, const FLOAT* boxcenter, const FLOAT beta, const FLOAT* los) {
  // Returns the residual, r=f-Av, keeping track of factors of h = boxsize/nmesh
  // First compute the operator A on v, keeping track of periodic
  // boundary conditions and ignoring the 1/h terms.
  // Note the relative signs here and in jacobi (or gauss_seidel).
  const size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  const int nmeshz = nmesh[2];
  const int nmeshyz = nmesh[2]*nmesh[1];
  FLOAT cell, cell2[NDIM], icell2[NDIM], offset[NDIM], losn[NDIM];
  for (int idim=0; idim<NDIM; idim++) {
    cell = boxsize[idim]/nmesh[idim];
    cell2[idim] = cell*cell;
    icell2[idim] = 1./cell2[idim];
    offset[idim] = (boxcenter[idim] - boxsize[idim]/2.)/cell;
    if (los != NULL) losn[idim] = los[idim]/cell;
  }
  #pragma omp parallel for shared(v,r)
  for (int ix=0; ix<nmesh[0]; ix++) {
    FLOAT px = (los == NULL) ? ix + offset[0] : losn[0];
    size_t ix0 = nmeshyz*ix;
    size_t ixp = nmeshyz*((ix+1) % nmesh[0]);
    size_t ixm = nmeshyz*((ix-1+nmesh[0]) % nmesh[0]);
    for (int iy=0; iy<nmesh[1]; iy++) {
      FLOAT py = (los == NULL) ? iy + offset[1] : losn[1];
      size_t iy0 = nmeshz*iy;
      size_t iyp = nmeshz*((iy+1) % nmesh[1]);
      size_t iym = nmeshz*((iy-1+nmesh[1]) % nmesh[1]);
      for (int iz0=0; iz0<nmesh[2]; iz0++) {
        FLOAT pz = (los == NULL) ? iz0 + offset[2] : losn[2];
        FLOAT g = beta/(cell2[0]*px*px+cell2[1]*py*py+cell2[2]*pz*pz);
        FLOAT gpx2 = icell2[0] + g*px*px;
        FLOAT gpy2 = icell2[1] + g*py*py;
        FLOAT gpz2 = icell2[2] + g*pz*pz;
        size_t izp = (iz0+1) % nmesh[2];
        size_t izm = (iz0-1+nmesh[2]) % nmesh[2];
        size_t ii = ix0 + iy0 + iz0;
        r[ii] = 2*(gpx2 + gpy2 + gpz2)*v[ii] -
               (gpx2*(v[ixp+iy0+iz0]+v[ixm+iy0+iz0])+
                gpy2*(v[ix0+iyp+iz0]+v[ix0+iym+iz0])+
                gpz2*(v[ix0+iy0+izp]+v[ix0+iy0+izm])+
                g/2*(px*py*(v[ixp+iyp+iz0]+v[ixm+iym+iz0]
                            -v[ixm+iyp+iz0]-v[ixp+iym+iz0])+
                     px*pz*(v[ixp+iy0+izp]+v[ixm+iy0+izm]
                            -v[ixm+iy0+izp]-v[ixp+iy0+izm])+
                     py*pz*(v[ix0+iyp+izp]+v[ix0+iym+izm]
                            -v[ix0+iym+izp]-v[ix0+iyp+izm])));
        if (los == NULL) {
          r[ii] -= g*(px*(v[ixp+iy0+iz0]-v[ixm+iy0+iz0])+
                      py*(v[ix0+iyp+iz0]-v[ix0+iym+iz0])+
                      pz*(v[ix0+iy0+izp]-v[ix0+iy0+izm]));
        }
      }
    }
  }
  // Now subtract it from f
  #pragma omp parallel for shared(r,f)
  for (size_t ii=0; ii<size; ii++) r[ii] = f[ii] - r[ii];
}


void prolong(const FLOAT* v2h, FLOAT* v1h, const int* nmesh) {
  // Transfer a vector, v2h, from the coarse grid with spacing 2h to a
  // fine grid with spacing 1h using linear interpolation and periodic BC.
  // The length, N, is of the coarse-grid vector, v2h.
  // This is simple, linear interpolation in a cube.
  const int nmeshz = nmesh[2];
  const int nmeshyz = nmesh[2]*nmesh[1];
  const int nmesh2z = 2*nmesh[2];
  const int nmesh2yz = 4*nmesh[2]*nmesh[1];
  #pragma omp parallel for shared(v2h,v1h)
  for (int ix=0; ix<nmesh[0]; ix++) {
    int ix0 = nmeshyz*ix;
    int ixp = nmeshyz*((ix+1) % nmesh[0]);
    int i2x0 = nmesh2yz*2*ix;
    int i2xp = i2x0 + nmesh2yz;
    for (int iy=0; iy<nmesh[1]; iy++) {
      int iy0 = nmeshz*iy;
      int iyp = nmeshz*((iy+1) % nmesh[1]);
      int i2y0 = nmesh2z*2*iy;
      int i2yp = i2y0 + nmesh2z;
      for (int iz0=0; iz0<nmesh[2]; iz0++) {
        int izp = (iz0+1) % nmesh[2];
        int i2z0 = 2*iz0;
        int i2zp = i2z0 + 1;
        int ii0 = ix0+iy0+iz0;
        v1h[i2x0+i2y0+i2z0] = v2h[ii0];
        v1h[i2xp+i2y0+i2z0] = (v2h[ii0] + v2h[ixp+iy0+iz0])/2;
        v1h[i2x0+i2yp+i2z0] = (v2h[ii0] + v2h[ix0+iyp+iz0])/2;
        v1h[i2x0+i2y0+i2zp] = (v2h[ii0] + v2h[ix0+iy0+izp])/2;
        v1h[i2xp+i2yp+i2z0] = (v2h[ii0] + v2h[ixp+iy0+iz0]
                              + v2h[ix0+iyp+iz0] + v2h[ixp+iyp+iz0])/4;
        v1h[i2x0+i2yp+i2zp] = (v2h[ii0] + v2h[ix0+iyp+iz0]
                              + v2h[ix0+iy0+izp] + v2h[ix0+iyp+izp])/4;
        v1h[i2xp+i2y0+i2zp] = (v2h[ii0] + v2h[ixp+iy0+iz0]
                              + v2h[ix0+iy0+izp] + v2h[ixp+iy0+izp])/4;
        v1h[i2xp+i2yp+i2zp] = (v2h[ii0] + v2h[ixp+iy0+iz0]
                              + v2h[ix0+iyp+iz0] + v2h[ix0+iy0+izp]
                              + v2h[ixp+iyp+iz0] + v2h[ixp+iy0+izp]
                              + v2h[ix0+iyp+izp] + v2h[ixp+iyp+izp])/8;
      }
    }
  }
}


void reduce(const FLOAT* v1h, FLOAT* v2h, const int* nmesh) {
  // Transfer a vector, v1h, from the fine grid with spacing 1h to a coarse
  // grid with spacing 2h using full weighting and periodic BC.
  // The length, N, is of the fine-grid vector (v1h) and is assumed even,
  // the code doesn't check.
  const int nmeshz = nmesh[2];
  const int nmeshyz = nmesh[2]*nmesh[1];
  int nmesh2[NDIM];
  for (int idim=0; idim<NDIM; idim++) nmesh2[idim] = nmesh[idim]/2;
  const int nmesh2z = nmesh2[2];
  const int nmesh2yz = nmesh2[2]*nmesh2[1];
  #pragma omp parallel for shared(v2h,v1h)
  for (int ix=0; ix<nmesh2[0]; ix++) {
    size_t ix0 = nmeshyz*2*ix;
    size_t ixp = nmeshyz*((2*ix+1) % nmesh[0]);
    size_t ixm = nmeshyz*((2*ix-1 + nmesh[0]) % nmesh[0]);
    for (int iy=0; iy<nmesh2[1]; iy++) {
      size_t iy0 = nmeshz*2*iy;
      size_t iyp = nmeshz*((2*iy+1) % nmesh[1]);
      size_t iym = nmeshz*((2*iy-1 + nmesh[1]) % nmesh[1]);
      for (int iz=0; iz<nmesh2[2]; iz++) {
        size_t iz0 = 2*iz;
        size_t izp = (iz0+1) % nmesh[2];
        size_t izm = (iz0-1 + nmesh[2]) % nmesh[2];
        v2h[nmesh2yz*ix+nmesh2z*iy+iz] = (8*v1h[ix0+iy0+iz0]+
                                          4*(v1h[ixp+iy0+iz0]+
                                          v1h[ixm+iy0+iz0]+
                                          v1h[ix0+iyp+iz0]+
                                          v1h[ix0+iym+iz0]+
                                          v1h[ix0+iy0+izp]+
                                          v1h[ix0+iy0+izm])+
                                          2*(v1h[ixp+iyp+iz0]+
                                          v1h[ixm+iyp+iz0]+
                                          v1h[ixp+iym+iz0]+
                                          v1h[ixm+iym+iz0]+
                                          v1h[ixp+iy0+izp]+
                                          v1h[ixm+iy0+izp]+
                                          v1h[ixp+iy0+izm]+
                                          v1h[ixm+iy0+izm]+
                                          v1h[ix0+iyp+izp]+
                                          v1h[ix0+iym+izp]+
                                          v1h[ix0+iyp+izm]+
                                          v1h[ix0+iym+izm])+
                                          v1h[ixp+iyp+izp]+
                                          v1h[ixm+iyp+izp]+
                                          v1h[ixp+iym+izp]+
                                          v1h[ixm+iym+izp]+
                                          v1h[ixp+iyp+izm]+
                                          v1h[ixm+iyp+izm]+
                                          v1h[ixp+iym+izm]+
                                          v1h[ixm+iym+izm])/64.0;
      }
    }
  }
}



void vcycle(FLOAT* v, const FLOAT* f, const int* nmesh, const FLOAT* boxsize, const FLOAT* boxcenter, const FLOAT beta, const FLOAT damping_factor, const int niterations, const FLOAT* los) {
  // Does one V-cycle, with a recursive strategy, replacing v in the process.
  jacobi(v,f,nmesh,boxsize,boxcenter,beta,damping_factor,niterations,los);
  const size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  _Bool recurse = 1;
  for (int idim=0; idim<NDIM; idim++) recurse &= (nmesh[idim] > 4 && (nmesh[idim] % 2 == 0));
  if (recurse) {
    // Not at coarsest level -- recurse coarser.
    int nmesh2[NDIM];
    for (int idim=0; idim<NDIM; idim++) nmesh2[idim] = nmesh[idim]/2;
    FLOAT* r = (FLOAT *) malloc(size*sizeof(FLOAT));
    //FLOAT* r = (FLOAT *) calloc(size,sizeof(FLOAT));
    residual(v,f,r,nmesh,boxsize,boxcenter,beta,los);
    FLOAT* f2h = (FLOAT *) malloc(size/8*sizeof(FLOAT));
    reduce(r,f2h,nmesh);
    free(r);
    // Make a vector of zeros as our first guess.
    FLOAT* v2h = (FLOAT *) calloc(size/8,sizeof(FLOAT));
    // and recursively call ourself
    vcycle(v2h,f2h,nmesh2,boxsize,boxcenter,beta,damping_factor,niterations,los);
    free(f2h);
    // take the residual and prolong it back to the finer grid
    FLOAT* v1h = (FLOAT *) malloc(size*sizeof(FLOAT));
    prolong(v2h,v1h,nmesh2);
    free(v2h);
    // and correct our earlier guess.
    for (size_t ii=0; ii<size; ii++) v[ii] += v1h[ii];
    free(v1h);
  }
  jacobi(v,f,nmesh,boxsize,boxcenter,beta,damping_factor,niterations,los);
}


FLOAT* fmg(FLOAT* f1h, FLOAT* v1h, const int* nmesh, const FLOAT* boxsize, const FLOAT* boxcenter, const FLOAT beta,
          const FLOAT jacobi_damping_factor, const int jacobi_niterations, const int vcycle_niterations, const FLOAT* los) {
  // The full multigrid cycle, also done recursively.
  //printf("NUMTHREADS %d\n", get_num_threads());
  const size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  _Bool recurse = 1;
  for (int idim=0; idim<NDIM; idim++) recurse &= (nmesh[idim] > 4 && (nmesh[idim] % 2 == 0));
  if (recurse) {
    // Recurse to a coarser grid.
    int nmesh2[NDIM];
    for (int idim=0; idim<NDIM; idim++) nmesh2[idim] = nmesh[idim]/2;
    FLOAT* f2h = (FLOAT *) malloc(size/8*sizeof(FLOAT));
    reduce(f1h,f2h,nmesh);
    FLOAT *v2h = fmg(f2h,NULL,nmesh2,boxsize,boxcenter,beta,jacobi_damping_factor,jacobi_niterations,vcycle_niterations,los);
    free(f2h);
    if (v1h == NULL) v1h = (FLOAT *) calloc(size,sizeof(FLOAT));
    prolong(v2h,v1h,nmesh2);
    free(v2h);
  }
  else {
    // Start with a guess of zero
    if (v1h == NULL) v1h = (FLOAT *) calloc(size,sizeof(FLOAT));
  }
  for (int iter=0; iter<vcycle_niterations; iter++) vcycle(v1h,f1h,nmesh,boxsize,boxcenter,beta,jacobi_damping_factor,jacobi_niterations,los);
  return v1h;
}
