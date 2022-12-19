import numpy as np
from pmesh.pm import ParticleMesh

from pyrecon.utils import MemoryMonitor


def test_mesh():
    with MemoryMonitor() as mem:
        nmesh = [256] * 3
        pm = ParticleMesh(BoxSize=[1.] * 3, Nmesh=nmesh)
        mem('init')
        v = np.zeros(shape=nmesh, dtype='f8')
        mesh = pm.create('real', value=v)
        v[...] = 1.
        mem('create')


if __name__ == '__main__':

    test_mesh()
