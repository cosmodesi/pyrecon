import numpy as np
from pmesh.pm import ParticleMesh

from pyrecon.utils import MemoryMonitor


def test_mesh():
    nmesh = 4
    pm = ParticleMesh(BoxSize=[1. * nmesh] * 3, Nmesh=[nmesh] * 3, dtype='f8')
    field = pm.create('real', value=0)
    field = pm.paint(0.1 * np.array([nmesh] * 3)[None, :], resampler='cic', hold=True, out=field)
    print(field.value)

    with MemoryMonitor() as mem:
        nmesh = [256] * 3
        pm = ParticleMesh(BoxSize=[1.] * 3, Nmesh=nmesh)
        mem('init')
        v = np.zeros(shape=nmesh, dtype='f8')
        mesh = pm.create('real', value=v)
        v[...] = 1.
        mem('create')

    nmesh = 8
    pm = ParticleMesh(BoxSize=[1.] * 3, Nmesh=[nmesh] * 3, dtype='c16')
    field = pm.create('complex')
    ik = []
    for iik in field.i:
        iik = np.ravel(iik)
        iik[iik >= nmesh // 2] -= nmesh
        ik.append(iik)
    print(ik)


if __name__ == '__main__':

    test_mesh()
