import os
import sys
import subprocess

import numpy as np
import fitsio

from pyrecon import PlaneParallelFFTReconstruction


def test_plane_parallel_fft(data_fn, randoms_fn):
    boxsize = 1200.
    boxcenter = [1754, 400, 400]
    data = fitsio.read(data_fn)
    randoms = fitsio.read(randoms_fn)
    data = {name: data[name] for name in data.dtype.names}
    randoms = {name: randoms[name] for name in randoms.dtype.names}
    recon = PlaneParallelFFTReconstruction(f=0.8,bias=2.,los='x',nthreads=4,boxcenter=boxcenter,boxsize=boxsize,nmesh=128,dtype='f8')
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])
    recon.set_density_contrast()
    recon.run()

    data['Position_rec'] = data['Position'] - recon.read_shifts(data['Position'])
    randoms['Position_rec'] = randoms['Position'] - recon.read_shifts(randoms['Position'], field='disp')

    from matplotlib import pyplot as plt
    from nbodykit.lab import ArrayCatalog, FKPCatalog, ConvolvedFFTPower
    data = ArrayCatalog(data)
    randoms = ArrayCatalog(randoms)

    for catalog in [data,randoms]:
        catalog['WEIGHT_FKP'] = np.ones(catalog.size,dtype='f8')
        catalog['WEIGHT_COMP'] = catalog['Weight']

    fkp = FKPCatalog(data, randoms)
    BoxSize = 1000.
    Nmesh = 128
    ells = (0, 2)
    mesh = fkp.to_mesh(position='Position',fkp_weight='WEIGHT_FKP',comp_weight='WEIGHT_COMP',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True,compensated=True)
    power = ConvolvedFFTPower(mesh,poles=ells,kmin=0.,dk=0.01)

    mesh = fkp.to_mesh(position='Position_rec',fkp_weight='WEIGHT_FKP',comp_weight='WEIGHT_COMP',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True,compensated=True)
    power_rec = ConvolvedFFTPower(mesh,poles=ells,kmin=0.,dk=0.01)

    for ill,ell in enumerate(ells):
        pk = power.poles['power_{:d}'.format(ell)] - power.attrs['shotnoise'] if ell == 0 else power.poles['power_{:d}'.format(ell)]
        plt.plot(power.poles['k'],power.poles['k']*pk,color='C{:d}'.format(ill),linestyle='-')
        pk = power_rec.poles['power_{:d}'.format(ell)] - power_rec.attrs['shotnoise'] if ell == 0 else power_rec.poles['power_{:d}'.format(ell)]
        plt.plot(power_rec.poles['k'],power_rec.poles['k']*pk,color='C{:d}'.format(ill),linestyle='--')

    plt.show()


if __name__ == '__main__':

    import utils
    from utils import data_fn, randoms_fn
    from pyrecon.utils import setup_logging

    setup_logging()
    test_plane_parallel_fft(data_fn, randoms_fn)
