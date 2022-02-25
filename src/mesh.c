#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utils.h"


void set_num_threads(int num_threads)
{
  if (num_threads>0) omp_set_num_threads(num_threads);
}


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


int assign_cic(FLOAT* mesh, const int* nmesh, const FLOAT* positions, const FLOAT* weights, size_t npositions) {
  // Assign positions (weights) to mesh
  // Asumes periodic boundaries
  // Positions must be in [0,nmesh-1]
  const size_t nmeshz = nmesh[2];
  const size_t nmeshyz = nmesh[2]*nmesh[1];
  for (size_t ii=0; ii<npositions; ii++) {
    const FLOAT weight = weights[ii];
    const FLOAT *pos = &(positions[ii*NDIM]);
    int ix0 = ((int) pos[0]) % nmesh[0];
    int iy0 = ((int) pos[1]) % nmesh[1];
    int iz0 = ((int) pos[2]) % nmesh[2];
    //int ix0 = (int) pos[0];
    //int iy0 = (int) pos[1];
    //int iz0 = (int) pos[2];
    //if (ix0<0 || ix0>=nmesh[0] || iy0<0 || iy0>=nmesh[1] || iz0<0 || iz0>=nmesh[2]) {
    //  printf("Index out of range: (ix,iy,iz) = (%d,%d,%d) for (%.3f,%.3f,%.3f)\n",ix0,iy0,iz0,pos[0],pos[1],pos[2]);
    //  return -1;
    //}
    FLOAT dx = pos[0] - ix0;
    FLOAT dy = pos[1] - iy0;
    FLOAT dz = pos[2] - iz0;
    size_t ixp = nmeshyz*((ix0+1) % nmesh[0]);
    size_t iyp = nmeshz*((iy0+1) % nmesh[1]);
    size_t izp = (iz0+1) % nmesh[2];
    ix0 *= nmeshyz;
    iy0 *= nmeshz;
    mesh[ix0+iy0+iz0] += (1-dx)*(1-dy)*(1-dz)*weight;
    mesh[ix0+iy0+izp] += (1-dx)*(1-dy)*dz*weight;
    mesh[ix0+iyp+iz0] += (1-dx)*dy*(1-dz)*weight;
    mesh[ix0+iyp+izp] += (1-dx)*dy*dz*weight;
    mesh[ixp+iy0+iz0] += dx*(1-dy)*(1-dz)*weight;
    mesh[ixp+iy0+izp] += dx*(1-dy)*dz*weight;
    mesh[ixp+iyp+iz0] += dx*dy*(1-dz)*weight;
    mesh[ixp+iyp+izp] += dx*dy*dz*weight;
  }
  return 0;
}


int convolve(FLOAT* mesh, const int* nmesh, const FLOAT* kernel, const int* nkernel) {
  // Performs a Gaussian smoothing using brute-force convolution.
  // Asumes periodic boundaries
  FLOAT sumw = 0;
  size_t size = nkernel[0]*nkernel[1]*nkernel[2];
  const size_t nkernelz = nkernel[2];
  const size_t nkernelyz = nkernel[2]*nkernel[1];
  for (size_t ii=0; ii<size; ii++) sumw += kernel[ii];
  // Take a copy of the data we're smoothing.
  size = nmesh[0]*nmesh[1]*nmesh[2];
  const size_t nmeshz = nmesh[2];
  const size_t nmeshyz = nmesh[2]*nmesh[1];
  FLOAT *ss = (FLOAT *) malloc(size*sizeof(FLOAT));
  for (size_t ii=0; ii<size; ii++) ss[ii] = mesh[ii];
  FLOAT rad[NDIM];
  for (int idim=0; idim<NDIM; idim++) {
    rad[idim] = nkernel[idim] / 2;
    if (nkernel[idim] % 2 == 0) {
      printf("Kernel size must be odd");
      free(ss);
      return -1;
    }
  }
  #pragma omp parallel for shared(mesh,ss,kernel)
  for (int ix=0; ix<nmesh[0]; ix++) {
    for (int iy=0; iy<nmesh[1]; iy++) {
      for (int iz=0; iz<nmesh[2]; iz++) {
        FLOAT sumd = 0;
        for (int dx=-rad[0]; dx<=rad[0]; dx++) {
          size_t iix = nmeshyz*((ix+dx+nmesh[0]) % nmesh[0]);
          size_t jjx = nkernelyz*(rad[0]+dx);
          for (int dy=-rad[1]; dy<=rad[1]; dy++) {
            size_t iiy = nmeshz*((iy+dy+nmesh[1]) % nmesh[1]);
            size_t jjy = nkernelz*(rad[1]+dy);
            for (int dz=-rad[2]; dz<=rad[2]; dz++) {
              int iiz = (iz+dz+nmesh[2]) % nmesh[2];
              size_t ii = iix+iiy+iiz;
              size_t jj = jjx+jjy+rad[2]+dz;
              sumd += kernel[jj]*ss[ii];
            }
          }
        }
        mesh[nmeshyz*ix+nmeshz*iy+iz] = sumd/sumw;
      }
    }
  }
  free(ss);
  return 0;
}


int smooth_gaussian(FLOAT* mesh, const int* nmesh, const FLOAT* smoothing_radius, const FLOAT nsigmas) {
  // Performs a Gaussian smoothing using brute-force convolution.
  // Now set up the smoothing stencil.
  // The number of grid points to search: >= nsigmas * smoothing_radius.
  int rad[NDIM], nkernel[NDIM];
  FLOAT fact[NDIM];
  for (int idim=0; idim<NDIM; idim++) {
    rad[idim] = (size_t) (nsigmas*smoothing_radius[idim]*nmesh[idim] + 1.0);
    nkernel[idim] = 2*rad[idim] + 1;
    fact[idim] = 1.0/(nmesh[idim]*smoothing_radius[idim])/(nmesh[idim]*smoothing_radius[idim]);
  }
  FLOAT *kernel = (FLOAT *) malloc(nkernel[0]*nkernel[1]*nkernel[2]*sizeof(FLOAT));
  #pragma omp parallel for shared(kernel)
  for (int dx=-rad[0]; dx<=rad[0]; dx++) {
    for (int dy=-rad[1]; dy<=rad[1]; dy++) {
      for (int dz=-rad[2]; dz<=rad[2]; dz++) {
        size_t ii = nkernel[1]*nkernel[2]*(rad[0]+dx)+nkernel[0]*(dy+rad[1])+(dz+rad[2]);
        FLOAT r2 = fact[0]*dx*dx+fact[1]*dy*dy+fact[2]*dz*dz;
        if (r2 < nsigmas*nsigmas) kernel[ii] = exp(-r2/2);
        else kernel[ii] = 0;
      }
    }
  }
  convolve(mesh,nmesh,kernel,nkernel);
  free(kernel);
  return 0;
}

/*
void smooth_fft_gaussian(FLOAT *mesh, const int* nmesh, const FLOAT* smoothing_radius) {
  // Performs a Gaussian smoothing using the FFTW library (v3), assumed to be
  // installed already. Rf is assumed to be in box units.
  // Make temporary vectors. FFTW uses double precision.
  size_t size = nmesh[0]*nmesh[1]*nmesh[2];
  fftw_complex * meshk = (fftw_complex *) malloc(nmesh[0]*nmesh[1]*(nmesh[2]/2+1)*sizeof(fftw_complex));
  // Generate the FFTW plan files.
  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());
  fftw_plan fplan = fftw_plan_dft_r2c_3d(nmesh[0],nmesh[1],nmesh[2],mesh,meshk,FFTW_ESTIMATE);
  fftw_plan iplan = fftw_plan_dft_c2r_3d(nmesh[0],nmesh[1],nmesh[2],meshk,mesh,FFTW_ESTIMATE);
  fftw_execute(fplan);
  // Now multiply by the smoothing filter.
  FLOAT fact[NDIM];
  for (int idim=0; idim<NDIM; idim++) fact[idim] = 0.5*smoothing_radius[idim]*smoothing_radius[idim]*(2*M_PI)*(2*M_PI);
  #pragma omp parallel for shared(meshk)
  for (int ix=0; ix<nmesh[0]; ix++) {
    int iix = (ix<=nmesh[0]/2) ? ix : ix-nmesh[0];
    for (int iy=0; iy<nmesh[1]; iy++) {
      int iiy = (iy<=nmesh[1]/2) ? iy : iy-nmesh[1];
      for (int iz=0; iz<nmesh[2]/2+1; iz++) {
        int iiz = iz;
        size_t ip = nmesh[1]*(nmesh[2]/2+1)*ix+(nmesh[2]/2+1)*iy+iz;
        FLOAT smth = exp(-fact[0]*iix*iix+fact[1]*iiy*iiy+fact[2]*iiz*iiz));
        meshk[ip][0] *= smth;
        meshk[ip][1] *= smth;
      }
    }
  }
  meshk[0][0] = meshk[0][1] = 0;	// Set the mean to zero.
  fftw_execute(iplan);
  #pragma omp parallel for shared(mesh)
  for (size_t ii=0; ii<size; ii++) mesh[ii] /= size;
  fftw_destroy_plan(fplan);
  fftw_destroy_plan(iplan);
  fftw_cleanup_threads();
  free(meshk);
}
*/


int read_finite_difference_cic(const FLOAT* mesh, const int* nmesh, const FLOAT* boxsize, const FLOAT* positions, FLOAT* shifts, size_t npositions) {
  // Computes the displacement field from mesh using second-order accurate
  // finite difference and shifts the data and randoms.
  // The displacements are pulled from the grid onto the positions of the
  // particles using CIC.
  // Positions must be in [0,nmesh-1]
  // Output is in boxsize unit
  const size_t nmeshz = nmesh[2];
  const size_t nmeshyz = nmesh[2]*nmesh[1];
  FLOAT cell[NDIM];
  for (int idim=0; idim<NDIM; idim++) cell[idim] = 2.0*boxsize[idim]/nmesh[idim];
  int flag = 0;
  #pragma omp parallel for shared(mesh,positions,shifts)
  for (size_t ii=0; ii<npositions; ii++) {
    if (flag) continue;
    // This is written out in gory detail both to make it easier to
    // see what's going on and to encourage the compiler to optimize
    // and vectorize the code as much as possible.
    const FLOAT *pos = &(positions[ii*NDIM]);
    int ix0 = ((int) pos[0]) % nmesh[0];
    int iy0 = ((int) pos[1]) % nmesh[1];
    int iz0 = ((int) pos[2]) % nmesh[2];
    //int ix0 = (int) pos[0];
    //int iy0 = (int) pos[1];
    //int iz0 = (int) pos[2];
    //if (ix0<0 || ix0>=nmesh[0] || iy0<0 || iy0>=nmesh[1] || iz0<0 || iz0>=nmesh[2]) {
    //  printf("Index out of range: (ix,iy,iz) = (%d,%d,%d) for (%.3f,%.3f,%.3f)\n",ix0,iy0,iz0,pos[0],pos[1],pos[2]);
    //  flag = 1;
    //  continue;
    //}
    FLOAT dx = pos[0] - ix0;
    FLOAT dy = pos[1] - iy0;
    FLOAT dz = pos[2] - iz0;
    size_t ixp = nmeshyz*((ix0+1) % nmesh[0]);
    size_t ixpp = nmeshyz*((ix0+2) % nmesh[0]);
    size_t ixm = nmeshyz*((ix0-1+nmesh[0]) % nmesh[0]);
    size_t iyp = nmeshz*((iy0+1) % nmesh[1]);
    size_t iypp = nmeshz*((iy0+2) % nmesh[1]);
    size_t iym = nmeshz*((iy0-1+nmesh[1]) % nmesh[1]);
    size_t izp = (iz0+1) % nmesh[2];
    size_t izpp = (iz0+2) % nmesh[2];
    size_t izm = (iz0-1+nmesh[2]) % nmesh[2];
    ix0 *= nmeshyz;
    iy0 *= nmeshz;
    FLOAT px,py,pz,wt;
    wt = (1-dx)*(1-dy)*(1-dz);
    px = (mesh[ixp+iy0+iz0]-mesh[ixm+iy0+iz0])*wt;
    py = (mesh[ix0+iyp+iz0]-mesh[ix0+iym+iz0])*wt;
    pz = (mesh[ix0+iy0+izp]-mesh[ix0+iy0+izm])*wt;
    wt = dx*(1-dy)*(1-dz);
    px += (mesh[ixpp+iy0+iz0]-mesh[ix0+iy0+iz0])*wt;
    py += (mesh[ixp+iyp+iz0]-mesh[ixp+iym+iz0])*wt;
    pz += (mesh[ixp+iy0+izp]-mesh[ixp+iy0+izm])*wt;
    wt = (1-dx)*dy*(1-dz);
    px += (mesh[ixp+iyp+iz0]-mesh[ixm+iyp+iz0])*wt;
    py += (mesh[ix0+iypp+iz0]-mesh[ix0+iy0+iz0])*wt;
    pz += (mesh[ix0+iyp+izp]-mesh[ix0+iyp+izm])*wt;
    wt = (1-dx)*(1-dy)*dz;
    px += (mesh[ixp+iy0+izp]-mesh[ixm+iy0+izp])*wt;
    py += (mesh[ix0+iyp+izp]-mesh[ix0+iym+izp])*wt;
    pz += (mesh[ix0+iy0+izpp]-mesh[ix0+iy0+iz0])*wt;
    wt = dx*dy*(1-dz);
    px += (mesh[ixpp+iyp+iz0]-mesh[ix0+iyp+iz0])*wt;
    py += (mesh[ixp+iypp+iz0]-mesh[ixp+iy0+iz0])*wt;
    pz += (mesh[ixp+iyp+izp]-mesh[ixp+iyp+izm])*wt;
    wt = dx*(1-dy)*dz;
    px += (mesh[ixpp+iy0+izp]-mesh[ix0+iy0+izp])*wt;
    py += (mesh[ixp+iyp+izp]-mesh[ixp+iym+izp])*wt;
    pz += (mesh[ixp+iy0+izpp]-mesh[ixp+iy0+iz0])*wt;
    wt = (1-dx)*dy*dz;
    px += (mesh[ixp+iyp+izp]-mesh[ixm+iyp+izp])*wt;
    py += (mesh[ix0+iypp+izp]-mesh[ix0+iy0+izp])*wt;
    pz += (mesh[ix0+iyp+izpp]-mesh[ix0+iyp+iz0])*wt;
    wt = dx*dy*dz;
    px += (mesh[ixpp+iyp+izp]-mesh[ix0+iyp+izp])*wt;
    py += (mesh[ixp+iypp+izp]-mesh[ixp+iy0+izp])*wt;
    pz += (mesh[ixp+iyp+izpp]-mesh[ixp+iyp+iz0])*wt;
    FLOAT *sh = &(shifts[ii*NDIM]);
    //px *= boxsize[0]*boxsize[0];
    //py *= boxsize[0]*boxsize[0];
    //pz *= boxsize[0]*boxsize[0];
    sh[0] = px/cell[0];
    sh[1] = py/cell[1];
    sh[2] = pz/cell[2];
  }
  if (flag) return -1;
  return 0;
}


int read_cic(const FLOAT* mesh, const int* nmesh, const FLOAT* positions, FLOAT* shifts, size_t npositions) {
  // Positions must be in [0,nmesh-1]
  const size_t nmeshz = nmesh[2];
  const size_t nmeshyz = nmesh[2]*nmesh[1];
  int flag = 0;
  #pragma omp parallel for shared(mesh,positions,shifts,flag)
  for (size_t ii=0; ii<npositions; ii++) {
    if (flag) continue;
    const FLOAT *pos = &(positions[ii*NDIM]);
    int ix0 = ((int) pos[0]) % nmesh[0];
    int iy0 = ((int) pos[1]) % nmesh[1];
    int iz0 = ((int) pos[2]) % nmesh[2];
    //int ix0 = (int) pos[0];
    //int iy0 = (int) pos[1];
    //int iz0 = (int) pos[2];
    //if (ix0<0 || ix0>=nmesh[0] || iy0<0 || iy0>=nmesh[1] || iz0<0 || iz0>=nmesh[2]) {
    //  printf("Index out of range: (ix,iy,iz) = (%d,%d,%d) for (%.3f,%.3f,%.3f)\n",ix0,iy0,iz0,pos[0],pos[1],pos[2]);
    //  flag = 1;
    //  continue;
    //}
    FLOAT dx = pos[0] - ix0;
    FLOAT dy = pos[1] - iy0;
    FLOAT dz = pos[2] - iz0;
    size_t ixp = nmeshyz*((ix0+1) % nmesh[0]);
    size_t iyp = nmeshz*((iy0+1) % nmesh[1]);
    size_t izp = (iz0+1) % nmesh[2];
    ix0 *= nmeshyz;
    iy0 *= nmeshz;
    FLOAT px;
    px = mesh[ix0+iy0+iz0]*(1-dx)*(1-dy)*(1-dz);
    px += mesh[ix0+iy0+izp]*(1-dx)*(1-dy)*dz;
    px += mesh[ix0+iyp+iz0]*(1-dx)*dy*(1-dz);
    px += mesh[ix0+iyp+izp]*(1-dx)*dy*dz;
    px += mesh[ixp+iy0+iz0]*dx*(1-dy)*(1-dz);
    px += mesh[ixp+iy0+izp]*dx*(1-dy)*dz;
    px += mesh[ixp+iyp+iz0]*dx*dy*(1-dz);
    px += mesh[ixp+iyp+izp]*dx*dy*dz;
    shifts[ii] = px;
  }
  if (flag) return -1;
  return 0;
}

/*
int copy(FLOAT* input_array, FLOAT* output_array, const size_t size) {
  #pragma omp parallel for schedule(dynamic) shared(input_array, output_array)
  for (size_t ii=0; ii<size; ii++) output_array[ii] = input_array[ii];
  return 0;
}
*/
int copy(FLOAT* input_array, FLOAT* output_array, const size_t size) {
  int chunksize = 100000;
  #pragma omp parallel for schedule(static, chunksize) shared(input_array, output_array)
  for (size_t ii=0; ii<size; ii++) output_array[ii] = input_array[ii];
  return 0;
}


int prod_sum(FLOAT* mesh, const int* nmesh, const FLOAT* coords, const int exp) {
  // We expand everything to help compiler
  // Slightly faster than a numpy code
  // NOTE: coords should list arrays to apply along z, y and x, in this order
  const size_t nmeshz = nmesh[2];
  const size_t nmeshypz = nmesh[1] + nmesh[2];
  const size_t nmeshyz = nmesh[2]*nmesh[1];
  if (exp == -1) {
    #pragma omp parallel for shared(mesh)
    for (int ix=0; ix<nmesh[0]; ix++) {
      for (int iy=0; iy<nmesh[1]; iy++) {
        FLOAT xy = coords[nmeshz + iy] + coords[nmeshypz + ix];
        size_t ixy = nmeshyz*ix + nmeshz*iy;
        for (int iz=0; iz<nmesh[2]; iz++) mesh[ixy + iz] /= (xy + coords[iz]);
      }
    }
  }
  else if (exp == 1) {
    #pragma omp parallel for shared(mesh)
    for (int ix=0; ix<nmesh[0]; ix++) {
      for (int iy=0; iy<nmesh[1]; iy++) {
        FLOAT xy = coords[nmeshz + iy] + coords[nmeshypz + ix];
        size_t ixy = nmeshyz*ix + nmeshz*iy;
        for (int iz=0; iz<nmesh[2]; iz++) mesh[ixy + iz] *= (xy + coords[iz]);
      }
    }
  }
  else {
    #pragma omp parallel for shared(mesh)
    for (int ix=0; ix<nmesh[0]; ix++) {
      for (int iy=0; iy<nmesh[1]; iy++) {
        FLOAT xy = coords[nmeshz + iy] + coords[nmeshypz + ix];
        size_t ixy = nmeshyz*ix + nmeshz*iy;
        for (int iz=0; iz<nmesh[2]; iz++) mesh[ixy + iz] *= POW((xy + coords[iz]), exp);
      }
    }
  }
  return 0.;
}
