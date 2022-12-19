#from cython cimport floating
import numpy as np
cimport numpy as np


cdef extern from "_multigrid_imp.h":

    void _jacobi_float(float *v, const float *f, const size_t* nmesh, const size_t localnmeshx, const int offsetx, const double* boxsize, const double* boxcenter,
                       const double beta, const double damping_factor, const double* los)

    void _jacobi_double(double *v, const double *f, const size_t* nmesh, const size_t localnmeshx, const int offsetx, const double* boxsize, const double* boxcenter,
                       const double beta, const double damping_factor, const double* los)

    void _residual_float(const float* v, const float* f, float* r, const size_t* nmesh, const size_t localnmeshx, const int offsetx, const double* boxsize, const double* boxcenter,
                         const double beta, const double* los)

    void _residual_double(const double* v, const double* f, double* r, const size_t* nmesh, const size_t localnmeshx, const int offsetx, const double* boxsize, const double* boxcenter,
                          const double beta, const double* los)

    void _prolong_float(const float* v2h, float* v1h, const size_t* nmesh, const size_t localnmeshx, const int offsetx)

    void _prolong_double(const double* v2h, double* v1h, const size_t* nmesh, const size_t localnmeshx, const int offsetx)

    void _reduce_float(const float* v1h, float* v2h, const size_t* nmesh, const size_t localnmeshx, const int offsetx)

    void _reduce_double(const double* v1h, double* v2h, const size_t* nmesh, const size_t localnmeshx, const int offsetx)

    void _read_finite_difference_cic_float(const float* mesh, const size_t* nmesh, const size_t localnmeshx, const double* boxsize, const float* positions, float* shifts, size_t npositions)

    void _read_finite_difference_cic_double(const double* mesh, const size_t* nmesh, const size_t localnmeshx, const double* boxsize, const double* positions, double* shifts, size_t npositions)


def cslice(mesh, cstart, cstop, concatenate=True):
    mpicomm = mesh.pm.comm
    cstart_out = mpicomm.allgather(cstart < 0)
    if any(cstart_out):
        cstart1, cstop1 = cstart, cstop
        cstart2, cstop2 = 0, 0
        if cstart_out[mpicomm.rank]:
            cstart1, cstop1 = cstart + mesh.pm.Nmesh[0], mesh.pm.Nmesh[0]
            cstart2, cstop2 = 0, cstop
        toret = cslice(mesh, cstart1, cstop1, concatenate=False)
        toret += cslice(mesh, cstart2, cstop2, concatenate=False)
        if concatenate: toret = np.concatenate(toret, axis=0)
        return toret
    cstop_out = mpicomm.allgather(cstop > mesh.pm.Nmesh[0] and cstop > cstart)  # as above test may call with cstop = cstart
    if any(cstop_out):
        cstart1, cstop1 = 0, 0
        cstart2, cstop2 = cstart, cstop
        if cstop_out[mpicomm.rank]:
            cstart1, cstop1 = cstart, mesh.pm.Nmesh[0]
            cstart2, cstop2 = 0, cstop - mesh.pm.Nmesh[0]
        toret = cslice(mesh, cstart1, cstop1, concatenate=False)
        toret += cslice(mesh, cstart2, cstop2, concatenate=False)
        if concatenate: toret = np.concatenate(toret, axis=0)
        return toret

    mpicomm = mesh.pm.comm
    ranges = mpicomm.allgather((mesh.start[0], mesh.start[0] + mesh.shape[0]))
    argsort = np.argsort([start for start, stop in ranges])
    # Send requested slices
    sizes, all_slices = [], []
    for irank, (start, stop) in enumerate(ranges):
        lstart, lstop = max(cstart - start, 0), min(max(cstop - start, 0), stop - start)
        sizes.append(max(lstop - lstart, 0))
        all_slices.append(mpicomm.allgather(slice(lstart, lstop)))
    assert sum(sizes) == cstop - cstart
    toret = []
    for root in range(mpicomm.size):
        if mpicomm.rank == root:
            for rank in range(mpicomm.size):
                sl = all_slices[root][rank]
                if rank == root:
                    tmp = mesh.value[sl]
                else:
                    mpicomm.Send(np.ascontiguousarray(mesh.value[sl]), dest=rank, tag=43)
                    #mpi.send(mesh.value[all_slices[root][irank]], dest=irank, tag=44, mpicomm=mpicomm)
        else:
            tmp = np.empty_like(mesh.value, shape=(sizes[root],) + mesh.shape[1:], order='C')
            mpicomm.Recv(tmp, source=root, tag=43)
            #tmp = mpi.recv(source=root, tag=44, mpicomm=mpicomm)
        toret.append(tmp)

    toret = [toret[ii] for ii in argsort]
    if concatenate: toret = np.concatenate(toret, axis=0)
    return toret


def jacobi(v, f, np.ndarray[double, ndim=1, mode='c'] boxcenter, double beta, double damping_factor=0.4, int niterations=5, double[:] los=None):
    dtype = v.dtype

    start, stop = v.start[0], v.start[0] + v.shape[0]
    cdef np.ndarray bv
    cdef np.ndarray bf = cslice(f, start - 1, stop + 1)

    cdef np.ndarray[size_t, ndim=1, mode='c'] nmesh = np.array(v.pm.Nmesh, dtype=np.uint64)
    cdef np.ndarray[double, ndim=1, mode='c'] boxsize = v.pm.BoxSize
    cdef double * plos = NULL
    if los is not None:
        plos = &los[0]
    cdef int offsetx = v.start[0] - 1

    for iter in range(niterations):
        bv = cslice(v, start - 1, stop + 1)
        if dtype.itemsize == 4:
            _jacobi_float(<float*> bv.data, <float*> bf.data, &nmesh[0], bv.shape[0], offsetx, &boxsize[0], &boxcenter[0], beta, damping_factor, plos)
        else:
            _jacobi_double(<double*> bv.data, <double*> bf.data, &nmesh[0], bv.shape[0], offsetx, &boxsize[0], &boxcenter[0], beta, damping_factor, plos)
        v.value = bv[1:-1]
    return v


def residual(v, f, np.ndarray[double, ndim=1, mode='c'] boxcenter, double beta, double[:] los=None):
    dtype = v.dtype

    start, stop = v.start[0], v.start[0] + v.shape[0]
    cdef np.ndarray bv = cslice(v, start - 1, stop + 1)
    cdef np.ndarray bf = cslice(f, start - 1, stop + 1)
    cdef np.ndarray br = np.empty_like(bv, order='C')

    cdef np.ndarray[size_t, ndim=1, mode='c'] nmesh = np.array(v.pm.Nmesh, dtype=np.uint64)
    cdef np.ndarray[double, ndim=1, mode='c'] boxsize = v.pm.BoxSize
    cdef double * plos = NULL
    if los is not None:
      plos = &los[0]
    cdef int offsetx = v.start[0] - 1

    if dtype.itemsize == 4:
        _residual_float(<float*> bv.data, <float*> bf.data, <float*> br.data, &nmesh[0], bv.shape[0], offsetx, &boxsize[0], &boxcenter[0], beta, plos)
    else:
        _residual_double(<double*> bv.data, <double*> bf.data, <double*> br.data, &nmesh[0], bv.shape[0], offsetx, &boxsize[0], &boxcenter[0], beta, plos)

    toret = v.pm.create(type='real')
    toret.value = br[1:-1]
    return toret


def prolong(v2h):
    dtype = v2h.dtype
    v1h = v2h.pm.reshape(v2h.pm.Nmesh * 2).create(type='real')
    start, stop = v1h.start[0], v1h.start[0] + v1h.shape[0]
    offsetx = int(v1h.start[0] % 2)
    cdef np.ndarray bv2h = cslice(v2h, start // 2, (stop - start + 1) // 2 + start // 2 + 1)
    cdef np.ndarray bv1h = np.ascontiguousarray(v1h.value)

    cdef np.ndarray[size_t, ndim=1, mode='c'] nmesh = np.array(v2h.pm.Nmesh, dtype=np.uint64)

    if dtype.itemsize == 4:
        _prolong_float(<float*> bv2h.data, <float*> bv1h.data, &nmesh[0], v1h.shape[0], offsetx)
    else:
        _prolong_double(<double*> bv2h.data, <double*> bv1h.data, &nmesh[0], v1h.shape[0], offsetx)

    v1h.value = bv1h
    return v1h


def reduce(v1h):
    dtype = v1h.dtype
    v2h = v1h.pm.reshape(v1h.pm.Nmesh // 2).create(type='real')
    start, stop = v2h.start[0], v2h.start[0] + v2h.shape[0]
    cdef np.ndarray bv1h = cslice(v1h, 2*start - 1, 2*stop)
    cdef np.ndarray bv2h = np.ascontiguousarray(v2h.value)

    cdef np.ndarray[size_t, ndim=1, mode='c'] nmesh = np.array(v1h.pm.Nmesh, dtype=np.uint64)

    if dtype.itemsize == 4:
        _reduce_float(<float*> bv1h.data, <float*> bv2h.data, &nmesh[0], v2h.shape[0], 1)
    else:
        _reduce_double(<double*> bv1h.data, <double*> bv2h.data, &nmesh[0], v2h.shape[0], 1)

    v2h.value = bv2h
    return v2h


def read_finite_difference_cic(mesh, np.ndarray positions, np.ndarray[double, ndim=1, mode='c'] boxcenter):
    dtype = mesh.dtype
    rdtype = positions.dtype
    offset = boxcenter - mesh.pm.BoxSize / 2.
    positions = (positions - offset) % mesh.pm.BoxSize
    cellsize = mesh.pm.BoxSize / mesh.pm.Nmesh
    layout = mesh.pm.decompose(positions, smoothing=4)
    positions = np.ascontiguousarray(layout.exchange(positions) / cellsize, dtype=dtype)
    positions[:, 0] = (positions[:, 0] - mesh.start[0]) % mesh.pm.Nmesh[0]

    cdef np.ndarray v = np.ascontiguousarray(mesh.value)
    cdef np.ndarray values = np.empty_like(positions)
    cdef np.ndarray[size_t, ndim=1, mode='c'] nmesh = np.array(mesh.pm.Nmesh, dtype=np.uint64)
    cdef np.ndarray[double, ndim=1, mode='c'] boxsize = mesh.pm.BoxSize

    if dtype.itemsize == 4:
        _read_finite_difference_cic_float(<float*> v.data, &nmesh[0], mesh.shape[0], &boxsize[0], <float*> positions.data, <float*> values.data, len(positions))
    else:
        _read_finite_difference_cic_double(<double*> v.data, &nmesh[0], mesh.shape[0], &boxsize[0], <double*> positions.data, <double*> values.data, len(positions))

    return layout.gather(values, mode='sum', out=None).astype(rdtype)
