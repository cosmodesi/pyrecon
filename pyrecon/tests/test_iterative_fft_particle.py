import os
import sys
import subprocess
import time
import importlib

import numpy as np
import fitsio

from pyrecon.iterative_fft_particle import OriginalIterativeFFTParticleReconstruction, IterativeFFTParticleReconstruction
from pyrecon.utils import distance
from test_multigrid import get_random_catalog



def test_no_nrandoms():
    boxsize = 1000.
    data = get_random_catalog(boxsize=boxsize,seed=42)
    recon = IterativeFFTParticleReconstruction(f=0.8,bias=2.,los='x',nthreads=4,boxcenter=boxsize/2.,boxsize=boxsize,nmesh=8,dtype='f8')
    recon.assign_data(data['Position'],data['Weight'])
    assert not recon.has_randoms
    recon.set_density_contrast()
    assert np.allclose(np.mean(recon.mesh_delta), 0.)
    recon.run()
    assert np.all(np.abs(recon.read_shifts(data['Position'])) < 5.)


def test_dtype():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=84)
    for los in [None, 'x']:
        recon_f4 = IterativeFFTParticleReconstruction(f=0.8,bias=2.,nthreads=4,positions=randoms['Position'],nmesh=64,los=los,dtype='f4')
        recon_f4.assign_data(data['Position'],data['Weight'])
        recon_f4.assign_randoms(randoms['Position'],randoms['Weight'])
        recon_f4.set_density_contrast()
        assert recon_f4.mesh_delta.dtype.itemsize == 4
        recon_f4.run()
        assert recon_f4.mesh_psi[0].dtype.itemsize == 4
        shifts_f4 = recon_f4.read_shifts(data['Position'].astype('f8'),field='disp+rsd')
        assert shifts_f4.dtype.itemsize == 8
        shifts_f4 = recon_f4.read_shifts(data['Position'].astype('f4'),field='disp+rsd')
        assert shifts_f4.dtype.itemsize == 4
        recon_f8 = IterativeFFTParticleReconstruction(f=0.8,bias=2.,nthreads=4,positions=randoms['Position'],nmesh=64,los=los,dtype='f8')
        recon_f8.assign_data(data['Position'],data['Weight'])
        recon_f8.assign_randoms(randoms['Position'],randoms['Weight'])
        recon_f8.set_density_contrast()
        assert recon_f8.mesh_delta.dtype.itemsize == 8
        recon_f8.run()
        assert recon_f8.mesh_psi[0].dtype.itemsize == 8
        shifts_f8 = recon_f8.read_shifts(data['Position'],field='disp+rsd')
        assert shifts_f8.dtype.itemsize == 8
        assert not np.all(shifts_f4 == shifts_f8)
        assert np.allclose(shifts_f4, shifts_f8, atol=1e-2, rtol=1e-2)


def test_mem():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=84)
    from pyrecon.utils import MemoryMonitor
    with MemoryMonitor() as mem:
        recon = IterativeFFTParticleReconstruction(f=0.8,bias=2.,nthreads=4,positions=randoms['Position'],nmesh=256,dtype='f8')
        mem('init')
        recon.assign_data(data['Position'],data['Weight'])
        mem('data')
        recon.assign_randoms(randoms['Position'],randoms['Weight'])
        mem('randoms')
        recon.set_density_contrast()
        mem('delta')
        recon.run()
        mem('recon') # 3 meshes

def test_iterative_fft_particle_wrap():
    size = 100000
    boxsize = 1000
    for origin in [-500, 0, 500]:
        boxcenter = boxsize/2 + origin
        data = get_random_catalog(size, boxsize, seed=42)
        # set one of the data positions to be outside the fiducial box by hand
        data['Position'][-1] = np.array([boxsize, boxsize, boxsize]) + 1
        data['Position'] += boxcenter
        randoms = get_random_catalog(size, boxsize, seed=42)
        # set one of the random positions to be outside the fiducial box by hand
        randoms['Position'][-1] = np.array([0, 0, 0]) - 1
        randoms['Position'] += boxcenter
        recon = IterativeFFTParticleReconstruction(f=0.8, bias=2, los='z', boxsize=boxsize, boxcenter=boxcenter, nmesh=64, wrap=True)
        # following steps should run without error if wrapping is correctly implemented
        recon.assign_data(data['Position'],data['Weight'])
        recon.assign_randoms(randoms['Position'],randoms['Weight'])
        recon.set_density_contrast()
        recon.run()

        # following steps test the implementation coded into standalone pyrecon code
        for field in ['rsd', 'disp', 'disp+rsd']:
            shifts = recon.read_shifts('data', field=field)
            diff = data['Position'] - shifts
            positions_rec = (diff - recon.offset) % recon.boxsize + recon.offset
            assert np.all(positions_rec <= origin + boxsize) and np.all(positions_rec >= origin)

def test_los():
    boxsize = 1000.
    boxcenter = [boxsize/2]*3
    data = get_random_catalog(boxsize=boxsize,seed=42)
    randoms = get_random_catalog(boxsize=boxsize,seed=84)
    recon = IterativeFFTParticleReconstruction(f=0.8,bias=2.,los='x',nthreads=4,boxcenter=boxcenter,boxsize=boxsize,nmesh=64,dtype='f8')
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])
    recon.set_density_contrast()
    recon.run()
    shifts_global = recon.read_shifts(data['Position'],field='disp+rsd')
    offset = 1e8
    boxcenter[0] += offset
    data['Position'][:,0] += offset
    randoms['Position'][:,0] += offset
    recon = IterativeFFTParticleReconstruction(f=0.8,bias=2.,nthreads=4,boxcenter=boxcenter,boxsize=boxsize,nmesh=64,dtype='f8')
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])
    recon.set_density_contrast()
    recon.run()
    shifts_local = recon.read_shifts(data['Position'],field='disp+rsd')
    assert np.allclose(shifts_local,shifts_global,rtol=1e-3,atol=1e-3)


def test_iterative_fft(data_fn, randoms_fn):
    # here path to reference Julian's code: https://github.com/julianbautista/eboss_clustering/blob/master/python (python setup.py build_ext --inplace)
    sys.path.insert(0,'../../../../reconstruction/eboss_clustering/python')
    from cosmo import CosmoSimple
    from recon import Recon
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

    recon_ref = Recon(data['RA'],data['DEC'],data['Z'],data['Weight'],randoms['RA'],randoms['DEC'],randoms['Z'],randoms['Weight'],nbins=nmesh,smooth=smooth,f=f,bias=bias,padding=boxpad,nthreads=nthreads)
    data['Position'] = np.array([recon_ref.dat.x,recon_ref.dat.y,recon_ref.dat.z]).T
    randoms['Position'] = np.array([recon_ref.ran.x,recon_ref.ran.y,recon_ref.ran.z]).T
    for i in range(niter):
        recon_ref.iterate(i)
    recon_ref.apply_shifts_full()
    shifts_data_ref = np.array([getattr(recon_ref.dat,x) - getattr(recon_ref.dat,'new{}'.format(x)) for x in 'xyz']).T
    shifts_randoms_ref = np.array([getattr(recon_ref.ran,x) - getattr(recon_ref.ran,'new{}'.format(x)) for x in 'xyz']).T
    recon_ref.apply_shifts_rsd()
    recon_ref.apply_shifts_full()
    shifts_randoms_rsd_ref = np.array([getattr(recon_ref.ran,x) - getattr(recon_ref.ran,'new{}'.format(x)) for x in 'xyz']).T
    #recon_ref.summary()
    boxsize = recon_ref.binsize*recon_ref.nbins
    boxcenter = np.array([getattr(recon_ref,'{}min'.format(x)) for x in 'xyz']) + boxsize/2.

    print('')
    print('#'*50)
    print('')
    recon = OriginalIterativeFFTParticleReconstruction(f=recon_ref.f,bias=recon_ref.bias,boxsize=boxsize,boxcenter=boxcenter,nmesh=nmesh,fft_engine='numpy',nthreads=nthreads)
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])
    recon.set_density_contrast(smoothing_radius=smooth)
    recon.run(niterations=niter)
    shifts_data = recon.read_shifts('data',field='disp+rsd')
    shifts_randoms = recon.read_shifts(randoms['Position'],field='disp')
    shifts_randoms_rsd = recon.read_shifts(randoms['Position'],field='disp+rsd')
    #print(np.abs(np.diff(shifts_data-shifts_data_ref)).max(),np.abs(np.diff(shifts_randoms-shifts_randoms_ref)).max())
    print('abs test - ref',np.max(distance(shifts_data-shifts_data_ref)))
    print('rel test - ref',np.max(distance(shifts_data-shifts_data_ref)/distance(shifts_data_ref)))
    assert np.allclose(shifts_data,shifts_data_ref,rtol=1e-7,atol=1e-7)
    assert np.allclose(shifts_randoms,shifts_randoms_ref,rtol=1e-7,atol=1e-7)
    assert np.allclose(shifts_randoms_rsd,shifts_randoms_rsd_ref,rtol=1e-7,atol=1e-7)


def test_revolver(data_fn, randoms_fn=None):

    # here path to reference Julian's code: https://github.com/seshnadathur/Revolver (python python_tools/setup.py build_ext --inplace)
    sys.path.insert(0,'../../../../reconstruction/Revolver')
    spec = importlib.util.spec_from_file_location("name", '../../../../reconstruction/Revolver/parameters/default_params.py')
    parms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parms)
    parms.f = 0.8
    parms.bias = 1.4
    parms.verbose = False
    parms.nbins = 256
    isbox = parms.is_box = randoms_fn is None
    parms.nthreads = 4
    boxsize = 800.
    parms.box_length = boxsize
    niter = 3

    data = fitsio.read(data_fn)
    fields = ['Position','Weight']
    data = {field:data[field] for field in fields}

    if isbox:
        randoms = data
    else:
        randoms = fitsio.read(randoms_fn)
        randoms = {field:randoms[field] for field in fields}

    for catalog in [data, randoms]:
        for field in catalog:
            if catalog[field].dtype.byteorder == '>':
                catalog[field] = np.array(catalog[field].byteswap().newbyteorder(), dtype='f8')
        catalog['Distance'] = distance(catalog['Position'])

    class Catalog(object):

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    datacat = Catalog(**{axis: data['Position'][:,iaxis] for iaxis,axis in enumerate('xyz')},
                      **{'new{}'.format(axis): data['Position'][:,iaxis].copy() for iaxis,axis in enumerate('xyz')},
                      dist=data['Distance'], weight=data['Weight'], size=len(data['Position']), box_length=boxsize)

    rancat = Catalog(**{axis: randoms['Position'][:,iaxis] for iaxis,axis in enumerate('xyz')},
                     **{'new{}'.format(axis): randoms['Position'][:,iaxis].copy() for iaxis,axis in enumerate('xyz')},
                     dist=randoms['Distance'], weight=randoms['Weight'], size=len(randoms['Position']), box_length=boxsize)

    from python_tools.recon import Recon as Revolver
    t0 = time.time()
    recon_ref = Revolver(datacat, ran=rancat, parms=parms)
    #if isbox:
    #    recon_ref.ran = recon_ref.cat  # fudge to prevent error in call to apply_shifts_full
    for i in range(niter):
        recon_ref.iterate(i, debug=True)

    # save the full shifted version of the catalogue
    recon_ref.apply_shifts_full()
    print('Revolver completed in {:.2f} s'.format(time.time() - t0), flush=True)
    if isbox:
        shifts_data_ref = np.array([getattr(recon_ref.cat,x) - getattr(recon_ref.cat,'new{}'.format(x)) for x in 'xyz']).T % boxsize
        #shifts_data_ref = np.array([getattr(recon_ref.cat,'new{}'.format(x)) for x in 'xyz']).T
    else:
        shifts_data_ref = np.array([getattr(recon_ref.cat,x) - getattr(recon_ref.cat,'new{}'.format(x)) for x in 'xyz']).T
        shifts_randoms_ref = np.array([getattr(recon_ref.ran,x) - getattr(recon_ref.ran,'new{}'.format(x)) for x in 'xyz']).T

    if isbox:
        los = 'z'
        boxcenter = boxsize/2.
    else:
        los = None
        boxsize = recon_ref.binsize*recon_ref.nbins
        boxcenter = np.array([getattr(recon_ref,'{}min'.format(x)) for x in 'xyz']) + boxsize/2.

    def load_wisdom(wisdom_fn):
        import json
        with open(wisdom_fn, 'r') as file:
            wisdom = json.load(file)
            return tuple(w.encode('utf-8') for w in wisdom)

    wisdom_fn = 'wisdom.128.4'
    wisdom = load_wisdom(wisdom_fn)

    print('')
    print('#'*50)
    print('')
    t0 = time.time()
    recon = OriginalIterativeFFTParticleReconstruction(f=recon_ref.f,bias=recon_ref.bias,boxsize=boxsize,boxcenter=boxcenter,nmesh=recon_ref.nbins,los=los,
                                                       fft_engine='fftw',fft_wisdom=wisdom,nthreads=recon_ref.nthreads)
    #recon = OriginalIterativeFFTParticleReconstruction(f=recon_ref.f,bias=recon_ref.bias,boxsize=boxsize,boxcenter=boxcenter,nmesh=recon_ref.nbins,los=los,fft_engine='numpy',nthreads=recon_ref.nthreads)
    recon.assign_data(data['Position'],data['Weight'])
    if not isbox:
        recon.assign_randoms(randoms['Position'],randoms['Weight'])
    recon.set_density_contrast(smoothing_radius=recon_ref.smooth)
    recon.run(niterations=niter)
    if isbox:
        shifts_data = recon.read_shifts('data',field='disp+rsd') % boxsize
        #shifts_data = (data['Position'] - shifts_data) #% boxsize
    else:
        shifts_data = recon.read_shifts('data',field='disp+rsd')
        shifts_randoms = recon.read_shifts(randoms['Position'],field='disp')
    print('pyrecon completed in {:.2f} s'.format(time.time() - t0), flush=True)
    #print(np.abs(np.diff(shifts_data-shifts_data_ref)).max(),np.abs(np.diff(shifts_randoms-shifts_randoms_ref)).max())

    print(shifts_data_ref.min(), shifts_data_ref.max())
    print(shifts_data.min(), shifts_data.max())

    print('abs test - ref',np.max(distance(shifts_data-shifts_data_ref)))
    print('rel test - ref',np.max(distance(shifts_data-shifts_data_ref)/distance(shifts_data_ref)))
    assert np.allclose(shifts_data,shifts_data_ref,rtol=1e-7,atol=1e-7)
    if not parms.is_box:
        assert np.allclose(shifts_randoms,shifts_randoms_ref,rtol=1e-7,atol=1e-7)


def test_script(data_fn, randoms_fn, output_data_fn, output_randoms_fn):

    catalog_dir = '_catalogs'
    command = 'pyrecon config_iterativefft_particle.yaml --data-fn {} --randoms-fn {} --output-data-fn {} --output-randoms-fn {}'.format(
                os.path.relpath(data_fn,catalog_dir),os.path.relpath(randoms_fn,catalog_dir),
                os.path.relpath(output_data_fn,catalog_dir),os.path.relpath(output_randoms_fn,catalog_dir))
    subprocess.call(command,shell=True)
    data = fitsio.read(data_fn,columns=['Position','Weight'])
    randoms = fitsio.read(randoms_fn,columns=['Position','Weight'])
    recon = IterativeFFTParticleReconstruction(nthreads=4,positions=randoms['Position'],nmesh=128,dtype='f8')
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


def test_script_no_randoms(data_fn, output_data_fn):

    catalog_dir = '_catalogs'
    command = 'pyrecon config_iterativefft_particle_no_randoms.yaml --data-fn {} --output-data-fn {}'.format(
                os.path.relpath(data_fn,catalog_dir),os.path.relpath(output_data_fn,catalog_dir))
    subprocess.call(command,shell=True)
    data = fitsio.read(data_fn)
    boxsize = 800
    boxcenter = boxsize/2.
    recon = IterativeFFTParticleReconstruction(nthreads=4,los='x',boxcenter=boxcenter,boxsize=boxsize,nmesh=128,dtype='f8')
    recon.set_cosmo(f=0.8,bias=2.)
    recon.assign_data(data['RSDPosition'])
    recon.set_density_contrast()
    recon.run()

    ref_positions_rec_data = data['RSDPosition'] - recon.read_shifts('data')
    data = fitsio.read(output_data_fn,columns=['Position_rec'])
    assert np.allclose(ref_positions_rec_data,data['Position_rec'])


if __name__ == '__main__':

    import utils
    from utils import box_data_fn, data_fn, randoms_fn, catalog_dir
    from pyrecon.utils import setup_logging

    setup_logging()
    # Uncomment to compute catalogs needed for these tests
    #utils.setup()

    script_output_box_data_fn = os.path.join(catalog_dir,'script_box_data_rec.fits')
    script_output_data_fn = os.path.join(catalog_dir,'script_data_rec.fits')
    script_output_randoms_fn = os.path.join(catalog_dir,'script_randoms_rec.fits')

    #test_mem()
    test_no_nrandoms()
    test_dtype()
    test_los()
    test_iterative_fft(data_fn,randoms_fn)
    test_revolver(data_fn,randoms_fn)
    test_revolver(box_data_fn)
    test_script(data_fn,randoms_fn,script_output_data_fn,script_output_randoms_fn)
    test_script_no_randoms(box_data_fn, script_output_box_data_fn)
