import re

import numpy as np

from pyrecon import utils
from pyrecon.utils import DistanceToRedshift

from test_multigrid import get_random_catalog


def decode_eval_str(s):
    # change ${col} => col, and return list of columns
    toret = str(s)
    columns = []
    for replace in re.finditer(r'(\${.*?})', s):
        value = replace.group(1)
        col = value[2:-1]
        toret = toret.replace(value, col)
        if col not in columns: columns.append(col)
    return toret, columns


def test_decode_eval_str():
    s = '(${RA}>0.) & (${RA}<30.) & (${DEC}>0.) & (${DEC}<30.)'
    s, cols = decode_eval_str(s)
    print(s, cols)


def test_distance_to_redshift():

    def distance(z):
        return z**2

    d2z = DistanceToRedshift(distance)
    z = np.linspace(0., 20., 200)
    d = distance(z)
    assert np.allclose(d2z(d), z)
    for itemsize in [4, 8]:
        assert d2z(d.astype('f{:d}'.format(itemsize))).itemsize == itemsize


def test_cartesian_to_sky():
    for dtype in ['f4', 'f8']:
        dtype = np.dtype(dtype)
        positions = get_random_catalog(csize=100)['Position'].astype(dtype)
        drd = utils.cartesian_to_sky(positions)
        assert all(array.dtype.itemsize == dtype.itemsize for array in drd)
        positions2 = utils.sky_to_cartesian(*drd)
        assert positions2.dtype.itemsize == dtype.itemsize
        assert np.allclose(positions2, positions, rtol=1e-4 if dtype.itemsize == 4 else 1e-9)


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


def test_cslice():
    from pmesh.pm import ParticleMesh
    from pyrecon import mpi
    mpicomm = mpi.COMM_WORLD
    nmesh = 16
    pm = ParticleMesh(BoxSize=[1.] * 3, Nmesh=[nmesh] * 3, np=(mpicomm.size, 1), dtype='f8')
    mesh = pm.create('real')
    cvalue = np.arange(pm.Nmesh.prod()).reshape(pm.Nmesh)
    mesh.value[...] = cvalue[mesh.start[0]:mesh.start[0] + mesh.shape[0]]
    array = cslice(mesh, -1, nmesh + 1)
    assert array.shape == (nmesh + 2, nmesh, nmesh)
    assert np.allclose(array[0], cvalue[-1])
    assert np.allclose(array[-1], cvalue[0])
    array = cslice(mesh, -3, nmesh + 3)
    assert array.shape == (nmesh + 6, nmesh, nmesh)
    array = cslice(mesh, -9, nmesh + 3)
    assert array.shape == (nmesh + 12, nmesh, nmesh)
    array = cslice(mesh, 2, 6)
    assert array.shape == (4, nmesh, nmesh)
    sl = [(0, 3), (-1, 2), (1, 4), (2, 5)][mpicomm.rank]
    array = cslice(mesh, *sl)
    assert array[1:-1].shape == (1, nmesh, nmesh)


if __name__ == '__main__':

    test_decode_eval_str()
    test_distance_to_redshift()
    test_cartesian_to_sky()
    test_cslice()
