#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

// This is a readaptation of Martin J. White's code, available at https://github.com/martinjameswhite/recon_code
// C++ dependencies have been removed, solver parameters e.g. niterations exposed
// Grid can be non-cubic, with a cellsize different along each direction (but why would we want that?)
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


void mkname(_jacobi)(FLOAT *v, const FLOAT *f, const size_t* nmesh, const size_t localnmeshx, const int offsetx, const double* boxsize, const double* boxcenter,
                     const double beta, const double damping_factor, const double* los) {
  // Does an update using damped Jacobi. This, and in residual below,
  // is where the explicit equation we are solving appears.
  // See notes for more details.
  const size_t localsize = localnmeshx*nmesh[1]*nmesh[2];
  const size_t nmeshz = nmesh[2];
  const size_t nmeshyz = nmesh[2]*nmesh[1];
  FLOAT* jac = (FLOAT *) my_malloc(localsize, sizeof(FLOAT));
  FLOAT cellsize, cellsize2[NDIM], icellsize2[NDIM], offset[NDIM], losn[NDIM];
  for (int idim=0; idim<NDIM; idim++) {
    cellsize = boxsize[idim]/nmesh[idim];
    cellsize2[idim] = cellsize*cellsize;
    icellsize2[idim] = 1./cellsize2[idim];
    offset[idim] = (boxcenter[idim] - boxsize[idim]/2.)/cellsize;
    if (los != NULL) losn[idim] = los[idim]/cellsize;
  }
  for (size_t ix=0; ix<localnmeshx; ix++) {
    FLOAT px = (los == NULL) ? ix + offsetx + offset[0] : losn[0];
    size_t ix0 = nmeshyz*ix;
    size_t ixp = nmeshyz*((ix+1) % localnmeshx);
    size_t ixm = nmeshyz*((ix+localnmeshx-1) % localnmeshx);
#if defined(_OPENMP)
    #pragma omp parallel for shared(v, f, jac)
#endif
    for (size_t iy=0; iy<nmesh[1]; iy++) {
      FLOAT py = (los == NULL) ? iy + offset[1] : losn[1];
      size_t iy0 = nmeshz*iy;
      size_t iyp = nmeshz*((iy+1) % nmesh[1]);
      size_t iym = nmeshz*((iy+nmesh[1]-1) % nmesh[1]);
      for (size_t iz0=0; iz0<nmesh[2]; iz0++) {
        FLOAT pz = (los == NULL) ? iz0 + offset[2] : losn[2];
        FLOAT g = beta/(cellsize2[0]*px*px+cellsize2[1]*py*py+cellsize2[2]*pz*pz);
        FLOAT gpx2 = icellsize2[0] + g*px*px;
        FLOAT gpy2 = icellsize2[1] + g*py*py;
        FLOAT gpz2 = icellsize2[2] + g*pz*pz;
        size_t izp = (iz0+1) % nmesh[2];
        size_t izm = (iz0+nmesh[2]-1) % nmesh[2];
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
#if defined(_OPENMP)
  #pragma omp parallel for shared(v, jac)
#endif
  for (size_t ii=0; ii<localsize; ii++) v[ii] = (1-damping_factor)*v[ii] + damping_factor*jac[ii];
}


void mkname(_residual)(const FLOAT* v, const FLOAT* f, FLOAT* r, const size_t* nmesh, const size_t localnmeshx, const int offsetx, const double* boxsize, const double* boxcenter, const double beta, const double* los) {
  // Returns the residual, r=f-Av, keeping track of factors of h = boxsize/nmesh
  // First compute the operator A on v, keeping track of periodic
  // boundary conditions and ignoring the 1/h terms.
  // Note the relative signs here and in jacobi (or gauss_seidel).
  const size_t localsize = localnmeshx*nmesh[1]*nmesh[2];
  const size_t nmeshz = nmesh[2];
  const size_t nmeshyz = nmesh[2]*nmesh[1];
  FLOAT cell, cell2[NDIM], icell2[NDIM], offset[NDIM], losn[NDIM];
  for (int idim=0; idim<NDIM; idim++) {
    cell = boxsize[idim]/nmesh[idim];
    cell2[idim] = cell*cell;
    icell2[idim] = 1./cell2[idim];
    offset[idim] = (boxcenter[idim] - boxsize[idim]/2.)/cell;
    if (los != NULL) losn[idim] = los[idim]/cell;
  }
  for (size_t ix=0; ix<localnmeshx; ix++) {
    FLOAT px = (los == NULL) ? ix + offsetx + offset[0] : losn[0];
    size_t ix0 = nmeshyz*ix;
    size_t ixp = nmeshyz*((ix+1) % localnmeshx);
    size_t ixm = nmeshyz*((ix+localnmeshx-1) % localnmeshx);
#if defined(_OPENMP)
    #pragma omp parallel for shared(v, r)
#endif
    for (size_t iy=0; iy<nmesh[1]; iy++) {
      FLOAT py = (los == NULL) ? iy + offset[1] : losn[1];
      size_t iy0 = nmeshz*iy;
      size_t iyp = nmeshz*((iy+1) % nmesh[1]);
      size_t iym = nmeshz*((iy+nmesh[1]-1) % nmesh[1]);
      for (size_t iz0=0; iz0<nmesh[2]; iz0++) {
        FLOAT pz = (los == NULL) ? iz0 + offset[2] : losn[2];
        FLOAT g = beta/(cell2[0]*px*px+cell2[1]*py*py+cell2[2]*pz*pz);
        FLOAT gpx2 = icell2[0] + g*px*px;
        FLOAT gpy2 = icell2[1] + g*py*py;
        FLOAT gpz2 = icell2[2] + g*pz*pz;
        size_t izp = (iz0+1) % nmesh[2];
        size_t izm = (iz0+nmesh[2]-1) % nmesh[2];
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
#if defined(_OPENMP)
  #pragma omp parallel for shared(r, f)
#endif
  for (size_t ii=0; ii<localsize; ii++) r[ii] = f[ii] - r[ii];
}

void mkname(_prolong)(const FLOAT* v2h, FLOAT* v1h, const size_t* nmesh, const size_t localnmeshx, const int offsetx) {
  // Transfer a vector, v2h, from the coarse grid with spacing 2h to a
  // fine grid with spacing 1h using linear interpolation and periodic BC.
  // The length, N, is of the coarse-grid vector, v2h.
  // This is simple, linear interpolation in a cube.
  const size_t nmeshz = nmesh[2];
  const size_t nmeshyz = nmesh[2]*nmesh[1];
  const size_t nmesh1z = 2*nmesh[2];
  const size_t nmesh1yz = 4*nmesh[2]*nmesh[1];
  const size_t nmeshx = (localnmeshx+1)/2 + 1;
  for (size_t i1x=0; i1x<localnmeshx; i1x++) {
    size_t i1x0 = nmesh1yz*i1x;
    int i1xo = i1x + offsetx;
    size_t ix0 = nmeshyz*((i1xo/2) % nmeshx);
    size_t ixp = nmeshyz*((i1xo/2+1) % nmeshx);
#if defined(_OPENMP)
    #pragma omp parallel for shared(v2h, v1h)
#endif
    for (size_t iy=0; iy<nmesh[1]; iy++) {
      size_t iy0 = nmeshz*iy;
      size_t iyp = nmeshz*((iy+1) % nmesh[1]);
      size_t i1y0 = nmesh1z*2*iy;
      size_t i1yp = i1y0 + nmesh1z;
      for (size_t iz0=0; iz0<nmesh[2]; iz0++) {
        size_t izp = (iz0+1) % nmesh[2];
        size_t i1z0 = 2*iz0;
        size_t i1zp = i1z0+1;
        size_t ii0 = ix0+iy0+iz0;
        if (i1xo % 2 == 0) {
          v1h[i1x0+i1y0+i1z0] = v2h[ii0];
          v1h[i1x0+i1yp+i1z0] = (v2h[ii0] + v2h[ix0+iyp+iz0])/2;
          v1h[i1x0+i1y0+i1zp] = (v2h[ii0] + v2h[ix0+iy0+izp])/2;
          v1h[i1x0+i1yp+i1zp] = (v2h[ii0] + v2h[ix0+iyp+iz0]
                                + v2h[ix0+iy0+izp] + v2h[ix0+iyp+izp])/4;
        } else {
          v1h[i1x0+i1y0+i1z0] = (v2h[ii0] + v2h[ixp+iy0+iz0])/2;
          v1h[i1x0+i1yp+i1z0] = (v2h[ii0] + v2h[ixp+iy0+iz0]
                                + v2h[ix0+iyp+iz0] + v2h[ixp+iyp+iz0])/4;
          v1h[i1x0+i1y0+i1zp] = (v2h[ii0] + v2h[ixp+iy0+iz0]
                                + v2h[ix0+iy0+izp] + v2h[ixp+iy0+izp])/4;
          v1h[i1x0+i1yp+i1zp] = (v2h[ii0] + v2h[ixp+iy0+iz0]
                                + v2h[ix0+iyp+iz0] + v2h[ix0+iy0+izp]
                                + v2h[ixp+iyp+iz0] + v2h[ixp+iy0+izp]
                                + v2h[ix0+iyp+izp] + v2h[ixp+iyp+izp])/8;
        }
      }
    }
  }
}


void mkname(_reduce)(const FLOAT* v1h, FLOAT* v2h, const size_t* nmesh, const size_t localnmeshx, const int offsetx) {
  // Transfer a vector, v1h, from the fine grid with spacing 1h to a coarse
  // grid with spacing 2h using full weighting and periodic BC.
  // The length, N, is of the fine-grid vector (v1h) and is assumed even,
  // the code doesn't check.
  const size_t nmeshz = nmesh[2];
  const size_t nmeshyz = nmesh[2]*nmesh[1];
  size_t nmesh2[NDIM];
  for (int idim=0; idim<NDIM; idim++) nmesh2[idim] = nmesh[idim]/2;
  const size_t nmesh2z = nmesh2[2];
  const size_t nmesh2yz = nmesh2[2]*nmesh2[1];
  const size_t nmeshx = 2*localnmeshx + offsetx;
  for (size_t ix=0; ix<localnmeshx; ix++) {
    size_t ix0 = nmeshyz*(2*ix+offsetx);
    size_t ixp = nmeshyz*((2*ix+offsetx+1) % nmeshx);
    size_t ixm = nmeshyz*((2*ix+offsetx+nmeshx-1) % nmeshx);
#if defined(_OPENMP)
    #pragma omp parallel for shared(v2h, v1h)
#endif
    for (size_t iy=0; iy<nmesh2[1]; iy++) {
      size_t iy0 = nmeshz*2*iy;
      size_t iyp = nmeshz*((2*iy+1) % nmesh[1]);
      size_t iym = nmeshz*((2*iy+nmesh[1]-1) % nmesh[1]);
      for (size_t iz=0; iz<nmesh2[2]; iz++) {
        size_t iz0 = 2*iz;
        size_t izp = (iz0+1) % nmesh[2];
        size_t izm = (iz0+nmesh[2]-1) % nmesh[2];
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


int mkname(_read_finite_difference_cic)(const FLOAT* mesh, const size_t* nmesh, const size_t localnmeshx, const double* boxsize, const FLOAT* positions, FLOAT* shifts, size_t npositions) {
  // Computes the displacement field from mesh using second-order accurate
  // finite difference and shifts the data and randoms.
  // The displacements are pulled from the grid onto the positions of the
  // particles using CIC.
  // Input positions must be in [0, nmesh]
  // Output is in boxsize unit
  const size_t nmeshz = nmesh[2];
  const size_t nmeshyz = nmesh[2]*nmesh[1];
  FLOAT cellsize[NDIM];
  for (int idim=0; idim<NDIM; idim++) cellsize[idim] = 2.0*boxsize[idim]/nmesh[idim];
#if defined(_OPENMP)
  #pragma omp parallel for shared(mesh, positions, shifts)
#endif
  for (size_t ii=0; ii<npositions; ii++) {
    // This is written out in gory detail both to make it easier to
    // see what's going on and to encourage the compiler to optimize
    // and vectorize the code as much as possible.
    const FLOAT *pos = &(positions[ii*NDIM]);
    FLOAT dx = pos[0] - (int) pos[0];
    FLOAT dy = pos[1] - (int) pos[1];
    FLOAT dz = pos[2] - (int) pos[2];
    size_t ix0 = (int) pos[0];
    size_t lix0 = ix0;
    size_t lixp = (ix0+1) % nmesh[0];
    size_t lixpp = (ix0+2) % nmesh[0];
    size_t lixm = (ix0+nmesh[0]-1) % nmesh[0];
    size_t iy0 = ((int) pos[1]) % nmesh[1];
    size_t iz0 = ((int) pos[2]) % nmesh[2];
    size_t ixp = nmeshyz*lixp;
    size_t ixpp = nmeshyz*lixpp;
    size_t ixm = nmeshyz*lixm;
    size_t iyp = nmeshz*((iy0+1) % nmesh[1]);
    size_t iypp = nmeshz*((iy0+2) % nmesh[1]);
    size_t iym = nmeshz*((iy0+nmesh[1]-1) % nmesh[1]);
    size_t izp = (iz0+1) % nmesh[2];
    size_t izpp = (iz0+2) % nmesh[2];
    size_t izm = (iz0+nmesh[2]-1) % nmesh[2];
    ix0 *= nmeshyz;
    iy0 *= nmeshz;
    FLOAT px=0,py=0,pz=0,wt;
    if ((lix0 >= 0) && (lix0 < localnmeshx)) {
      wt = (1-dx)*(1-dy)*(1-dz);
      py += (mesh[ix0+iyp+iz0]-mesh[ix0+iym+iz0])*wt;
      pz += (mesh[ix0+iy0+izp]-mesh[ix0+iy0+izm])*wt;

      wt = (1-dx)*dy*(1-dz);
      py += (mesh[ix0+iypp+iz0]-mesh[ix0+iy0+iz0])*wt;
      pz += (mesh[ix0+iyp+izp]-mesh[ix0+iyp+izm])*wt;

      wt = (1-dx)*(1-dy)*dz;
      py += (mesh[ix0+iyp+izp]-mesh[ix0+iym+izp])*wt;
      pz += (mesh[ix0+iy0+izpp]-mesh[ix0+iy0+iz0])*wt;

      wt = (1-dx)*dy*dz;
      py += (mesh[ix0+iypp+izp]-mesh[ix0+iy0+izp])*wt;
      pz += (mesh[ix0+iyp+izpp]-mesh[ix0+iyp+iz0])*wt;

      wt = dx*(1-dy)*(1-dz);
      px -= mesh[ix0+iy0+iz0]*wt;

      wt = dx*dy*(1-dz);
      px -= mesh[ix0+iyp+iz0]*wt;

      wt = dx*(1-dy)*dz;
      px -= mesh[ix0+iy0+izp]*wt;

      wt = dx*dy*dz;
      px -= mesh[ix0+iyp+izp]*wt;
    }
    if ((lixm >= 0) && (lixm < localnmeshx)) {
      wt = (1-dx)*(1-dy)*(1-dz);
      px -= mesh[ixm+iy0+iz0]*wt;

      wt = (1-dx)*dy*(1-dz);
      px -= mesh[ixm+iyp+iz0]*wt;

      wt = (1-dx)*(1-dy)*dz;
      px -= mesh[ixm+iy0+izp]*wt;

      wt = (1-dx)*dy*dz;
      px -= mesh[ixm+iyp+izp]*wt;
    }
    if ((lixp >= 0) && (lixp < localnmeshx)) {
      wt = (1-dx)*(1-dy)*(1-dz);
      px += mesh[ixp+iy0+iz0]*wt;

      wt = (1-dx)*dy*(1-dz);
      px += mesh[ixp+iyp+iz0]*wt;

      wt = (1-dx)*(1-dy)*dz;
      px += mesh[ixp+iy0+izp]*wt;

      wt = (1-dx)*dy*dz;
      px += mesh[ixp+iyp+izp]*wt;

      wt = dx*(1-dy)*(1-dz);
      py += (mesh[ixp+iyp+iz0]-mesh[ixp+iym+iz0])*wt;
      pz += (mesh[ixp+iy0+izp]-mesh[ixp+iy0+izm])*wt;

      wt = dx*dy*(1-dz);
      py += (mesh[ixp+iypp+iz0]-mesh[ixp+iy0+iz0])*wt;
      pz += (mesh[ixp+iyp+izp]-mesh[ixp+iyp+izm])*wt;

      wt = dx*(1-dy)*dz;
      py += (mesh[ixp+iyp+izp]-mesh[ixp+iym+izp])*wt;
      pz += (mesh[ixp+iy0+izpp]-mesh[ixp+iy0+iz0])*wt;

      wt = dx*dy*dz;
      py += (mesh[ixp+iypp+izp]-mesh[ixp+iy0+izp])*wt;
      pz += (mesh[ixp+iyp+izpp]-mesh[ixp+iyp+iz0])*wt;
    }
    if ((lixpp >= 0) && (lixpp < localnmeshx)) {
      wt = dx*(1-dy)*(1-dz);
      px += mesh[ixpp+iy0+iz0]*wt;

      wt = dx*dy*(1-dz);
      px += mesh[ixpp+iyp+iz0]*wt;

      wt = dx*(1-dy)*dz;
      px += mesh[ixpp+iy0+izp]*wt;

      wt = dx*dy*dz;
      px += mesh[ixpp+iyp+izp]*wt;
    }
    FLOAT *sh = &(shifts[ii*NDIM]);
    sh[0] = px/cellsize[0];
    sh[1] = py/cellsize[1];
    sh[2] = pz/cellsize[2];
  }
  return 0;
}
