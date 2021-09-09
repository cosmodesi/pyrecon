import sys

import numpy as np
import fitsio

from pyrecon import IterativeFFTReconstruction

sys.path.insert(0,'../../../reconstruction/eboss_clustering/python')
import cosmo
from cosmo import CosmoSimple
from recon import Recon


def distance(pos):
    return np.sum(pos**2,axis=-1)**0.5

def test_iterative_fft(data_fn, randoms_fn):
    nmesh = 128
    smooth = 15.
    f = 0.81
    bias = 2.0
    Omega_m = 0.3
    boxpad = 200
    nthreads = 4
    niter = 3
    data = fitsio.read(data_fn)
    randoms = fitsio.read(randoms_fn)
    fields = ['Position','Weight']
    data = {field:data[field] for field in fields}
    randoms = {field:randoms[field] for field in fields}

    cosmo = CosmoSimple(omega_m=Omega_m)
    from pyrecon.utils import cartesian_to_sky
    for catalog in [data, randoms]:
        catalog['Z'],catalog['RA'],catalog['DEC'] = cartesian_to_sky(catalog['Position'])
        catalog['Z'] = cosmo.get_redshift(catalog['Z'])
        for field in catalog:
            if catalog[field].dtype.byteorder == '>':
                catalog[field] = catalog[field].byteswap().newbyteorder()

    rec_ref = Recon(data['RA'],data['DEC'],data['Z'],data['Weight'],randoms['RA'],randoms['DEC'],randoms['Z'],randoms['Weight'],nbins=nmesh,smooth=smooth,f=f,bias=bias,padding=boxpad,nthreads=nthreads)
    data['Position'] = np.array([rec_ref.dat.x,rec_ref.dat.y,rec_ref.dat.z]).T
    randoms['Position'] = np.array([rec_ref.ran.x,rec_ref.ran.y,rec_ref.ran.z]).T
    for i in range(niter):
        rec_ref.iterate(i)
    rec_ref.apply_shifts_full()
    shifts_data_ref = np.array([getattr(rec_ref.dat,x) - getattr(rec_ref.dat,'new{}'.format(x)) for x in 'xyz']).T
    shifts_randoms_ref = np.array([getattr(rec_ref.ran,x) - getattr(rec_ref.ran,'new{}'.format(x)) for x in 'xyz']).T
    #rec_ref.summary()
    boxsize = rec_ref.binsize*rec_ref.nbins
    boxcenter = np.array([getattr(rec_ref,'{}min'.format(x)) for x in 'xyz']) + boxsize/2.

    print('')
    print('#'*50)
    print('')
    rec = IterativeFFTReconstruction(f=rec_ref.f,bias=rec_ref.bias,boxsize=boxsize,boxcenter=boxcenter,nmesh=nmesh,fft_engine='numpy',nthreads=nthreads)
    rec.assign_data(data['Position'],data['Weight'])
    rec.assign_randoms(randoms['Position'],randoms['Weight'])
    rec.set_density_contrast()
    rec.run(niterations=niter,smoothing_radius=smooth)
    shifts_data = rec.read_shifts('data')
    shifts_randoms = rec.read_shifts(randoms['Position'],with_rsd=False)
    #print(np.abs(np.diff(shifts_data-shifts_data_ref)).max(),np.abs(np.diff(shifts_randoms-shifts_randoms_ref)).max())
    print('abs test - ref',np.max(distance(shifts_data-shifts_data_ref)))
    print('rel test - ref',np.max(distance(shifts_data-shifts_data_ref)/distance(shifts_data_ref)))


if __name__ == '__main__':

    import utils
    from utils import data_fn, randoms_fn, catalog_dir
    from pyrecon.utils import setup_logging
    setup_logging()
    #utils.setup()
    test_iterative_fft(data_fn,randoms_fn)
