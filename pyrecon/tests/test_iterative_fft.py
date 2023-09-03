import numpy as np

from pyrecon import IterativeFFTReconstruction
from pyrecon.utils import MemoryMonitor
from utils import get_random_catalog, Catalog, test_mpi


def test_mem():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=84)
    with MemoryMonitor() as mem:
        recon = IterativeFFTReconstruction(f=0.8, bias=2., positions=randoms['Position'], nmesh=256, dtype='f8')
        mem('init')
        recon.assign_data(data['Position'], data['Weight'])
        mem('data')
        recon.assign_randoms(randoms['Position'], randoms['Weight'])
        mem('randoms')
        recon.set_density_contrast()
        mem('delta')
        recon.run()
        mem('recon')  # 3 meshes


def test_dtype():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=81)
    for los in [None, 'x']:
        all_shifts = []
        for dtype in ['f4', 'f8']:
            dtype = np.dtype(dtype)
            itemsize = np.empty(0, dtype=dtype).real.dtype.itemsize
            recon = IterativeFFTReconstruction(f=0.8, bias=2., positions=randoms['Position'], nmesh=64, los=los, dtype=dtype)
            recon.assign_data(data['Position'], data['Weight'])
            recon.assign_randoms(randoms['Position'], randoms['Weight'])
            recon.set_density_contrast()
            assert recon.mesh_delta.dtype.itemsize == itemsize
            recon.run()
            assert recon.mesh_psi[0].dtype.itemsize == itemsize
            all_shifts2 = []
            for dtype2 in ['f4', 'f8']:
                dtype2 = np.dtype(dtype2)
                shifts = recon.read_shifts(data['Position'].astype(dtype2), field='disp+rsd')
                assert shifts.dtype.itemsize == dtype2.itemsize
                all_shifts2.append(shifts)
                if dtype2 == dtype: all_shifts.append(shifts)
            assert np.allclose(*all_shifts2, atol=1e-2, rtol=1e-2)


def test_wrap():
    size = 100000
    boxsize = 1000
    for boxcenter in [-500, 0, 500]:
        data = get_random_catalog(size, boxsize, seed=42)
        # set one of the data positions to be outside the fiducial box by hand
        data['Position'][-1] = np.array([boxsize, boxsize, boxsize]) + 1
        data['Position'] += boxcenter
        randoms = get_random_catalog(size, boxsize, seed=42)
        # set one of the random positions to be outside the fiducial box by hand
        randoms['Position'][-1] = np.array([0, 0, 0]) - 1
        randoms['Position'] += boxcenter
        recon = IterativeFFTReconstruction(f=0.8, bias=2, los='z', boxsize=boxsize, boxcenter=boxcenter, nmesh=64, wrap=True)
        # following steps should run without error if wrapping is correctly implemented
        recon.assign_data(data['Position'], data['Weight'])
        recon.assign_randoms(randoms['Position'], randoms['Weight'])
        recon.set_density_contrast()
        recon.run()

        # following steps test the implementation coded into standalone pyrecon code
        for field in ['rsd', 'disp', 'disp+rsd']:
            shifts = recon.read_shifts(data['Position'], field=field)
            diff = data['Position'] - shifts
            positions_rec = (diff - recon.offset) % recon.boxsize + recon.offset
            assert np.all(positions_rec >= boxcenter - boxsize / 2.) and np.all(positions_rec <= boxcenter + boxsize / 2.)
            assert np.allclose(recon.read_shifted_positions(data['Position'], field=field), positions_rec)


def test_ref(data_fn, randoms_fn, data_fn_rec=None, randoms_fn_rec=None):
    boxsize = 1200.
    boxcenter = [1754, 0, 0]
    data = Catalog.read(data_fn)
    randoms = Catalog.read(randoms_fn)
    recon = IterativeFFTReconstruction(f=0.8, bias=2., los=None, boxcenter=boxcenter, boxsize=boxsize, nmesh=128, dtype='f8')
    recon.assign_data(data['Position'], data['Weight'])
    recon.assign_randoms(randoms['Position'], randoms['Weight'])
    recon.set_density_contrast()
    recon.run(niterations=3)

    from pypower import CatalogFFTPower
    from matplotlib import pyplot as plt

    for cat, fn in zip([data, randoms], [data_fn_rec, randoms_fn_rec]):
        rec = recon.read_shifted_positions(cat['Position'])
        if 'Position_rec' in cat:
            if recon.mpicomm.rank == 0: print('Checking...')
            assert np.allclose(rec, cat['Position_rec'])
        else:
            cat['Position_rec'] = rec
        if fn is not None:
            cat.write(fn)

    kwargs = dict(edges={'min': 0., 'step': 0.01}, ells=(0, 2, 4), boxsize=1000., nmesh=64, resampler='tsc', interlacing=3, position_type='pos')
    power = CatalogFFTPower(data_positions1=data['Position'], randoms_positions1=randoms['Position'], **kwargs)
    poles = power.poles
    power = CatalogFFTPower(data_positions1=data['Position_rec'], randoms_positions1=randoms['Position_rec'], **kwargs)
    poles_rec = power.poles

    for ill, ell in enumerate(poles.ells):
        plt.plot(poles.k, poles.k * poles(ell=ell, complex=False), color='C{:d}'.format(ill), linestyle='-')
        plt.plot(poles_rec.k, poles_rec.k * poles_rec(ell=ell, complex=False), color='C{:d}'.format(ill), linestyle='--')

    if power.mpicomm.rank == 0:
        plt.show()


if __name__ == '__main__':

    from utils import data_fn, randoms_fn, catalog_rec_fn
    from pyrecon.utils import setup_logging

    setup_logging()
    # Run utils.py to generate catalogs needed for these tests

    # test_mem()
    test_dtype()
    test_wrap()
    test_mpi(IterativeFFTReconstruction)
    data_fn_rec, randoms_fn_rec = [catalog_rec_fn(fn, 'iterative_fft') for fn in [data_fn, randoms_fn]]
    # test_ref(data_fn, randoms_fn, data_fn_rec, randoms_fn_rec)
    test_ref(data_fn_rec, randoms_fn_rec, None, None)
