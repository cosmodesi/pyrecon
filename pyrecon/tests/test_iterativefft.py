import os
import sys
import subprocess

import numpy as np
import fitsio

from pyrecon.iterativefft import OriginalIterativeFFTReconstruction, IterativeFFTReconstruction
from pyrecon.utils import distance
from test_multigrid import get_random_catalog

# here path to reference Julian's code: https://github.com/julianbautista/eboss_clustering/blob/master/python
sys.path.insert(0,'../../../../reconstruction/eboss_clustering/python')
import cosmo
from cosmo import CosmoSimple
from recon import Recon


def test_los():
    boxsize = 1000.
    boxcenter = [boxsize/2]*3
    data = get_random_catalog(boxsize=boxsize,seed=42)
    randoms = get_random_catalog(boxsize=boxsize,seed=84)
    recon = IterativeFFTReconstruction(f=0.8,bias=2.,los='x',nthreads=4,boxcenter=boxcenter,boxsize=boxsize,nmesh=64,dtype='f8')
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])
    recon.set_density_contrast()
    recon.run()
    shifts_global = recon.read_shifts(data['Position'],with_rsd=True)
    offset = 1e8
    boxcenter[0] += offset
    data['Position'][:,0] += offset
    randoms['Position'][:,0] += offset
    recon = IterativeFFTReconstruction(f=0.8,bias=2.,nthreads=4,boxcenter=boxcenter,boxsize=boxsize,nmesh=64,dtype='f8')
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])
    recon.set_density_contrast()
    recon.run()
    shifts_local = recon.read_shifts(data['Position'],with_rsd=True)
    assert np.allclose(shifts_local,shifts_global,rtol=1e-3,atol=1e-3)


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
    rec = OriginalIterativeFFTReconstruction(f=rec_ref.f,bias=rec_ref.bias,boxsize=boxsize,boxcenter=boxcenter,nmesh=nmesh,fft_engine='numpy',nthreads=nthreads)
    rec.assign_data(data['Position'],data['Weight'])
    rec.assign_randoms(randoms['Position'],randoms['Weight'])
    rec.set_density_contrast()
    rec.run(niterations=niter,smoothing_radius=smooth)
    shifts_data = rec.read_shifts('data')
    shifts_randoms = rec.read_shifts(randoms['Position'],with_rsd=False)
    #print(np.abs(np.diff(shifts_data-shifts_data_ref)).max(),np.abs(np.diff(shifts_randoms-shifts_randoms_ref)).max())
    print('abs test - ref',np.max(distance(shifts_data-shifts_data_ref)))
    print('rel test - ref',np.max(distance(shifts_data-shifts_data_ref)/distance(shifts_data_ref)))


def test_script(data_fn, randoms_fn, output_data_fn, output_randoms_fn):

    catalog_dir = '_catalogs'
    command = 'pyrecon config_iterativefft.yaml --data-fn {} --randoms-fn {} --output-data-fn {} --output-randoms-fn {}'.format(
                os.path.relpath(data_fn,catalog_dir),os.path.relpath(randoms_fn,catalog_dir),
                os.path.relpath(script_output_data_fn,catalog_dir),os.path.relpath(script_output_randoms_fn,catalog_dir))
    subprocess.call(command,shell=True)
    data = fitsio.read(data_fn,columns=['Position','Weight'])
    randoms = fitsio.read(randoms_fn,columns=['Position','Weight'])
    recon = IterativeFFTReconstruction(nthreads=4,positions=randoms['Position'],nmesh=128,dtype='f8')
    recon.set_cosmo(f=0.8,bias=2.)
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])

    recon.set_density_contrast()
    recon.run()

    ref_positions_rec_data = data['Position'] - recon.read_shifts('data')
    ref_positions_rec_randoms = randoms['Position'] - recon.read_shifts(randoms['Position'])

    data = fitsio.read(output_data_fn,columns=['Position_rec'])
    randoms = fitsio.read(output_randoms_fn,columns=['Position_rec'])

    #print(ref_positions_rec_data,data['Position_rec'],ref_positions_rec_data-data['Position_rec'])
    assert np.allclose(ref_positions_rec_data,data['Position_rec'])
    assert np.allclose(ref_positions_rec_randoms,randoms['Position_rec'])


if __name__ == '__main__':

    import utils
    from utils import data_fn, randoms_fn, catalog_dir
    from pyrecon.utils import setup_logging

    setup_logging()
    test_los()
    # Uncomment to compute catalogs needed for these tests
    #utils.setup()

    script_output_data_fn = os.path.join(catalog_dir,'script_data_rec.fits')
    script_output_randoms_fn = os.path.join(catalog_dir,'script_randoms_rec.fits')

    test_iterative_fft(data_fn,randoms_fn)
    test_script(data_fn,randoms_fn,script_output_data_fn,script_output_randoms_fn)
