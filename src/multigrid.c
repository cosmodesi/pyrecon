#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//#include "multigrid.h"
#include "utils.h"

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

void jacobi(FLOAT *v, const FLOAT *f, const int* nmesh, const FLOAT* boxsize, const FLOAT* boxcenter, const FLOAT beta, const FLOAT damping_factor, const int niterations) {
  // Does an update using damped Jacobi. This, and in residual below,
  // is where the explicit equation we are solving appears.
  // See notes for more details.
  const size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  const int nmeshz = nmesh[2];
  const int nmeshyz = nmesh[2]*nmesh[1];
  FLOAT* jac = (FLOAT *) malloc(size*sizeof(FLOAT));
  FLOAT cell[NDIM], icell2=0;
  for (int idim=0; idim<NDIM; idim++) {
    cell[idim] = boxsize[idim]/nmesh[idim];
    icell2 += 1./(cell[idim]*cell[idim]);
  }
  for (int iter=0; iter<niterations; iter++) {
    //#pragma omp parallel for shared(v,f,jac)
    for (int ix=0; ix<nmesh[0]; ix++) {
      FLOAT px = boxcenter[0] + cell[0]*ix - boxsize[0]/2.;
      FLOAT rx = px/cell[0];
      size_t ix0 = nmeshyz*ix;
      size_t ixp = nmeshyz*((ix+1) % nmesh[0]);
      size_t ixm = nmeshyz*((ix-1+nmesh[0]) % nmesh[0]);
      for (int iy=0; iy<nmesh[1]; iy++) {
        FLOAT py = boxcenter[1] + cell[1]*iy - boxsize[1]/2.;
        FLOAT ry = py/cell[1];
        size_t iy0 = nmeshz*iy;
        size_t iyp = nmeshz*((iy+1) % nmesh[1]);
        size_t iym = nmeshz*((iy-1+nmesh[1]) % nmesh[1]);
        for (int iz=0; iz<nmesh[2]; iz++) {
          FLOAT pz = boxcenter[2] + cell[2]*iz - boxsize[2]/2.;
          FLOAT rz = pz/cell[2];
          FLOAT g = beta/(px*px+py*py+pz*pz);
          size_t iz0 = iz;
          size_t izp = (iz+1) % nmesh[2];
          size_t izm = (iz-1+nmesh[2]) % nmesh[2];
          size_t ii = ix0 + iy0 + iz0;
          jac[ii] = f[ii]+
                    (1+g*rx*rx)*(v[ixp+iy0+iz0]+v[ixm+iy0+iz0])+
                    (1+g*ry*ry)*(v[ix0+iyp+iz0]+v[ix0+iym+iz0])+
                    (1+g*rz*rz)*(v[ix0+iy0+izp]+v[ix0+iy0+izm])+
                    (g*rx*ry/2)*(v[ixp+iyp+iz0]+v[ixm+iym+iz0]
                                -v[ixm+iyp+iz0]-v[ixp+iym+iz0])+
                    (g*rx*rz/2)*(v[ixp+iy0+izp]+v[ixm+iy0+izm]
                                -v[ixm+iy0+izp]-v[ixp+iy0+izm])+
                    (g*ry*rz/2)*(v[ix0+iyp+izp]+v[ix0+iym+izm]
                                -v[ix0+iym+izp]-v[ix0+iyp+izm])+
                    (g*rx)*(v[ixp+iy0+iz0]-v[ixm+iy0+iz0])+
                    (g*ry)*(v[ix0+iyp+iz0]-v[ix0+iym+iz0])+
                    (g*rz)*(v[ix0+iy0+izp]-v[ix0+iy0+izm]);
          jac[ii] /= 2*(icell2 + g*(rx*rx + ry*ry + rz*rz));
          jac[ii] = f[ii];
        }
      }
    }
    #pragma omp parallel for shared(v,jac)
    for (size_t ii=0; ii<size; ii++) v[ii] = (1-damping_factor)*v[ii] + damping_factor*jac[ii];
  }
  free(jac);
}


void residual(const FLOAT* v, const FLOAT* f, FLOAT* r, const int* nmesh, const FLOAT* boxsize, const FLOAT* boxcenter, const FLOAT beta) {
  // Returns the residual, r=f-Av, keeping track of factors of h = boxsize/nmesh
  // First compute the operator A on v, keeping track of periodic
  // boundary conditions and ignoring the 1/h terms.
  // Note the relative signs here and in jacobi (or gauss_seidel).
  const size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  const int nmeshz = nmesh[2];
  const int nmeshyz = nmesh[2]*nmesh[1];
  FLOAT cell[NDIM], icell2=0;
  for (int idim=0; idim<NDIM; idim++) {
    cell[idim] = boxsize[idim]/nmesh[idim];
    icell2 += 1./(cell[idim]*cell[idim]);
  }
  //#pragma omp parallel for shared(v,r)
  for (int ix=0; ix<nmesh[0]; ix++) {
    FLOAT px = boxcenter[0] + cell[0]*ix - boxsize[0]/2.;
    FLOAT rx = px/cell[0];
    size_t ix0 = nmeshyz*ix;
    size_t ixp = nmeshyz*((ix+1) % nmesh[0]);
    size_t ixm = nmeshyz*((ix-1+nmesh[0]) % nmesh[0]);
    for (int iy=0; iy<nmesh[1]; iy++) {
      FLOAT py = boxcenter[1] + cell[1]*iy - boxsize[1]/2.;
      FLOAT ry = py/cell[1];
      size_t iy0 = nmeshz*iy;
      size_t iyp = nmeshz*((iy+1) % nmesh[1]);
      size_t iym = nmeshz*((iy-1+nmesh[1]) % nmesh[1]);
      for (int iz=0; iz<nmesh[2]; iz++) {
        FLOAT pz = boxcenter[2] + cell[2]*iz - boxsize[2]/2.;
        FLOAT rz = pz/cell[2];
        FLOAT g = beta/(px*px+py*py+pz*pz);
        size_t iz0 = iz;
        size_t izp = (iz+1) % nmesh[2];
        size_t izm = (iz-1+nmesh[2]) % nmesh[2];
        size_t ii = ix0 + iy0 + iz0;
        r[ii] = 2*(icell2 + g*(rx*rx + ry*ry + rz*rz))*v[ii] -
                  ((1+g*rx*rx)*(v[ixp+iy0+iz0]+v[ixm+iy0+iz0])+
                  (1+g*ry*ry)*(v[ix0+iyp+iz0]+v[ix0+iym+iz0])+
                  (1+g*rz*rz)*(v[ix0+iy0+izp]+v[ix0+iy0+izm])+
                  (g*rx*ry/2)*(v[ixp+iyp+iz0]+v[ixm+iym+iz0]
                              -v[ixm+iyp+iz0]-v[ixp+iym+iz0])+
                  (g*rx*rz/2)*(v[ixp+iy0+izp]+v[ixm+iy0+izm]
                              -v[ixm+iy0+izp]-v[ixp+iy0+izm])+
                  (g*ry*rz/2)*(v[ix0+iyp+izp]+v[ix0+iym+izm]
                              -v[ix0+iym+izp]-v[ix0+iyp+izm])+
                  (g*rx)*(v[ixp+iy0+iz0]-v[ixm+iy0+iz0])+
                  (g*ry)*(v[ix0+iyp+iz0]-v[ix0+iym+iz0])+
                  (g*rz)*(v[ix0+iy0+izp]-v[ix0+iy0+izm]));
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
  //#pragma omp parallel for shared(v2h,v1h)
  for (int ix=0; ix<nmesh[0]; ix++) {
    size_t ix0 = nmeshyz*ix;
    size_t ixp = nmeshyz*((ix+1) % nmesh[0]);
    size_t i2x0 = nmesh2yz*2*ix;
    size_t i2xp = nmesh2yz*(2*ix+1);
    for (int iy=0; iy<nmesh[1]; iy++) {
      size_t iy0 = nmeshz*iy;
      size_t iyp = nmeshz*((iy+1) % nmesh[1]);
      size_t i2y0 = nmesh2z*2*iy;
      size_t i2yp = nmesh2z*(2*iy+1);
      for (int iz=0; iz<nmesh[2]; iz++) {
        size_t iz0 = iz;
        size_t izp = (iz+1) % nmesh[2];
        size_t i2z0 = 2*iz;
        size_t i2zp = (2*iz + 1);
        size_t ii0 = ix0+iy0+iz0;
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
                              + v2h[ixp+iyp+iz0] + v2h[ix0+iyp+izp]
                              + v2h[ixp+iy0+izp] + v2h[ixp+iyp+izp])/8;
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
        size_t izp = (2*iz+1) % nmesh[2];
        size_t izm = (2*iz-1 + nmesh[2]) % nmesh[2];
        v2h[nmesh2yz*ix+nmesh2z*iy+iz] = (8*v1h[ix0+iy0+iz0]+
                                          4*v1h[ixp+iy0+iz0]+
                                          4*v1h[ixm+iy0+iz0]+
                                          4*v1h[ix0+iyp+iz0]+
                                          4*v1h[ix0+iym+iz0]+
                                          4*v1h[ix0+iy0+izp]+
                                          4*v1h[ix0+iy0+izm]+
                                          2*v1h[ixp+iyp+iz0]+
                                          2*v1h[ixm+iyp+iz0]+
                                          2*v1h[ixp+iym+iz0]+
                                          2*v1h[ixm+iym+iz0]+
                                          2*v1h[ixp+iy0+izp]+
                                          2*v1h[ixm+iy0+izp]+
                                          2*v1h[ixp+iy0+izm]+
                                          2*v1h[ixm+iy0+izm]+
                                          2*v1h[ix0+iyp+izp]+
                                          2*v1h[ix0+iym+izp]+
                                          2*v1h[ix0+iyp+izm]+
                                          2*v1h[ix0+iym+izm]+
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


void vcycle(FLOAT* v, const FLOAT* f, const int* nmesh, const FLOAT* boxsize, const FLOAT* boxcenter, const FLOAT beta, const FLOAT damping_factor, const int niterations) {
  // Does one V-cycle, with a recursive strategy, replacing v in the process.
  jacobi(v,f,nmesh,boxsize,boxcenter,beta,damping_factor,niterations);
  const size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  _Bool recurse = 1;
  for (int idim=0; idim<NDIM; idim++) recurse &= (nmesh[idim] > 4 && (nmesh[idim] % 2 == 0));
  if (recurse) {
    // Not at coarsest level -- recurse coarser.
    int nmesh2[NDIM];
    for (int idim=0; idim<NDIM; idim++) nmesh2[idim] = nmesh[idim]/2;
    //FLOAT* r = (FLOAT *) malloc(size*sizeof(FLOAT));
    FLOAT* r = (FLOAT *) calloc(size,sizeof(FLOAT));
    residual(v,f,r,nmesh,boxsize,boxcenter,beta);
    FLOAT* f2h = (FLOAT *) malloc(size/8*sizeof(FLOAT));
    reduce(r,f2h,nmesh);
    free(r);
    // Make a vector of zeros as our first guess.
    FLOAT* v2h = (FLOAT *) calloc(size/8,sizeof(FLOAT));
    // and recursively call ourself
    vcycle(v2h,f2h,nmesh2,boxsize,boxcenter,beta,damping_factor,niterations);
    free(f2h);
    // take the residual and prolong it back to the finer grid
    FLOAT* v1h = (FLOAT *) malloc(size*sizeof(FLOAT));
    prolong(v2h,v1h,nmesh2);
    free(v2h);
    // and correct our earlier guess.
    for (size_t ii=0; ii<size; ii++) v[ii] += v1h[ii];
    free(v1h);
  }
  jacobi(v,f,nmesh,boxsize,boxcenter,beta,damping_factor,niterations);
}

/*
void print_error(FLOAT* v, FLOAT* f, const int* nmesh,const FLOAT* boxsize, const FLOAT* boxcenter, const FLOAT beta) {
// For debugging purposes, prints an estimate of the residual.
  const size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  FLOAT* r = (FLOAT *) malloc(size*sizeof(FLOAT));
  residual(v,f,r,,nmesh,boxsize,boxcenter,beta);
  FLOAT res1=0,res2=0,src1=0,src2=0;
  #pragma omp parallel for shared(v,f) reduction(+:src1,src2,res1,res2)
  for (size_t ii=0; ii<size; ii++) {
    src1 += ABS(f[ii]);
    src2 += f[ii]*f[ii];
    res1 += ABS(r[ii]);
    res2 += r[ii]*r[ii];
  }
  free(r);
  src2 = SQRT(src2);
  res2 = SQRT(res2);
  printf("# Source L1 norm is %.7f\n",src1);
  printf("# Residual L1 norm is %.7f\n",res1);
  printf("# Source L2 norm is %.7f\n",src2);
  printf("# Residual L2 norm is %.7f\n",res2);
}
*/

/*
void fmg(const FLOAT* f1h, FLOAT* v1h, const int* nmesh, const FLOAT* boxsize, const FLOAT* boxcenter, const FLOAT beta,
          const FLOAT jacobi_damping_factor, const int jacobi_niterations, const int vcycle_niterations) {
  // The full multigrid cycle, also done recursively.
  const size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  _Bool recurse = 1;
  for (int idim=0; idim<NDIM; idim++) recurse &= ((nmesh[idim] > 4) && (nmesh[idim] % 2 == 0));
  if (recurse) {
    // Recurse to a coarser grid.
    int nmesh2[NDIM];
    for (int idim=0; idim<NDIM; idim++) nmesh2[idim] = nmesh[idim]/2;
    FLOAT* f2h = (FLOAT *) calloc(size/8,sizeof(FLOAT));
    reduce(f1h,f2h,nmesh);
    FLOAT* v2h = (FLOAT *) calloc(size,sizeof(FLOAT));
    fmg(f2h,v2h,nmesh2,boxsize,boxcenter,beta,jacobi_damping_factor,jacobi_niterations,vcycle_niterations);
    free(f2h);
    prolong(v2h,v1h,nmesh2);
    free(v2h);
  }
  for (int iter=0; iter<vcycle_niterations; iter++) vcycle(v1h,f1h,nmesh,boxsize,boxcenter,beta,jacobi_damping_factor,jacobi_niterations);
}
*/

FLOAT* fmg(FLOAT* f1h, FLOAT* v1h, const int* nmesh, const FLOAT* boxsize, const FLOAT* boxcenter, const FLOAT beta,
          const FLOAT jacobi_damping_factor, const int jacobi_niterations, const int vcycle_niterations) {
  // The full multigrid cycle, also done recursively.
  const size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  _Bool recurse = 1;
  for (int idim=0; idim<NDIM; idim++) recurse &= (nmesh[idim] > 4 && (nmesh[idim] % 2 == 0));
  if (recurse) {
    // Recurse to a coarser grid.
    int nmesh2[NDIM];
    for (int idim=0; idim<NDIM; idim++) nmesh2[idim] = nmesh[idim]/2;
    FLOAT* f2h = (FLOAT *) malloc(size/8*sizeof(FLOAT));
    reduce(f1h,f2h,nmesh);
    FLOAT *v2h = fmg(f2h,NULL,nmesh2,boxsize,boxcenter,beta,jacobi_damping_factor,jacobi_niterations,vcycle_niterations);
    free(f2h);
    if (v1h == NULL) v1h = (FLOAT *) calloc(size,sizeof(FLOAT));
    prolong(v2h,v1h,nmesh2);
    free(v2h);
  }
  else {
    // Start with a guess of zero
    if (v1h == NULL) v1h = (FLOAT *) calloc(size,sizeof(FLOAT));
  }
  for (int iter=0; iter<vcycle_niterations; iter++) vcycle(v1h,f1h,nmesh,boxsize,boxcenter,beta,jacobi_damping_factor,jacobi_niterations);
  return v1h;
}
