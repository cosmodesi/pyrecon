import numpy as np
import fitsio

from pyrecon import IterativeFFTReconstruction
from pyrecon.utils import MemoryMonitor
from test_multigrid import get_random_catalog


def test_dtype():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=81)
    for los in [None, 'x']:
        recon_f4 = IterativeFFTReconstruction(f=0.8, bias=2., nthreads=4, positions=randoms['Position'], nmesh=64, los=los, dtype='f4')
        recon_f4.assign_data(data['Position'], data['Weight'])
        recon_f4.assign_randoms(randoms['Position'], randoms['Weight'])
        recon_f4.set_density_contrast()
        assert recon_f4.mesh_delta.dtype.itemsize == 4
        recon_f4.run()
        assert recon_f4.mesh_psi[0].dtype.itemsize == 4
        shifts_f4 = recon_f4.read_shifts(data['Position'].astype('f8'), field='disp+rsd')
        assert shifts_f4.dtype.itemsize == 8
        shifts_f4 = recon_f4.read_shifts(data['Position'].astype('f4'), field='disp+rsd')
        assert shifts_f4.dtype.itemsize == 4
        recon_f8 = IterativeFFTReconstruction(f=0.8, bias=2., nthreads=4, positions=randoms['Position'], nmesh=64, los=los, dtype='f8')
        recon_f8.assign_data(data['Position'], data['Weight'])
        recon_f8.assign_randoms(randoms['Position'], randoms['Weight'])
        recon_f8.set_density_contrast()
        assert recon_f8.mesh_delta.dtype.itemsize == 8
        recon_f8.run()
        assert recon_f8.mesh_psi[0].dtype.itemsize == 8
        shifts_f8 = recon_f8.read_shifts(data['Position'], field='disp+rsd')
        assert shifts_f8.dtype.itemsize == 8
        assert not np.all(shifts_f4 == shifts_f8)
        assert np.allclose(shifts_f4, shifts_f8, atol=1e-2, rtol=1e-2)


def test_mem():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=84)
    with MemoryMonitor() as mem:
        recon = IterativeFFTReconstruction(f=0.8, bias=2., nthreads=4, positions=randoms['Position'], nmesh=256, dtype='f8')
        mem('init')
        recon.assign_data(data['Position'], data['Weight'])
        mem('data')
        recon.assign_randoms(randoms['Position'], randoms['Weight'])
        mem('randoms')
        recon.set_density_contrast()
        mem('delta')
        recon.run()
        mem('recon')  # 3 meshes


def test_iterative_fft_wrap():
    size = 100000
    boxsize = 1000
    for origin in [-500, 0, 500]:
        boxcenter = boxsize / 2 + origin
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
            assert np.all(positions_rec <= origin + boxsize) and np.all(positions_rec >= origin)
            assert np.allclose(recon.read_shifted_positions(data['Position'], field=field), positions_rec)


def test_iterative_fft(data_fn, randoms_fn):
    boxsize = 1200.
    boxcenter = [1754, 400, 400]
    data = fitsio.read(data_fn)
    randoms = fitsio.read(randoms_fn)
    data = {name: data[name] for name in data.dtype.names}
    randoms = {name: randoms[name] for name in randoms.dtype.names}
    recon = IterativeFFTReconstruction(f=0.8, bias=2., los=None, nthreads=4, boxcenter=boxcenter, boxsize=boxsize, nmesh=128, dtype='f8')
    recon.assign_data(data['Position'], data['Weight'])
    recon.assign_randoms(randoms['Position'], randoms['Weight'])
    recon.set_density_contrast()
    recon.run(niterations=3)

    from pypower import CatalogFFTPower
    from matplotlib import pyplot as plt

    data['Position_rec'] = data['Position'] - recon.read_shifts(data['Position'])
    randoms['Position_rec'] = randoms['Position'] - recon.read_shifts(randoms['Position'], field='disp')

    kwargs = dict(edges={'min': 0., 'step': 0.01}, ells=(0, 2, 4), boxsize=1000., nmesh=64, resampler='tsc', interlacing=3, position_type='pos')
    power = CatalogFFTPower(data_positions1=data['Position'], randoms_positions1=randoms['Position'], **kwargs)
    poles = power.poles
    power = CatalogFFTPower(data_positions1=data['Position_rec'], randoms_positions1=randoms['Position_rec'], **kwargs)
    poles_rec = power.poles

    for ill, ell in enumerate(poles.ells):
        plt.plot(poles.k, poles.k * poles(ell=ell), color='C{:d}'.format(ill), linestyle='-')
        plt.plot(poles_rec.k, poles_rec.k * poles_rec(ell=ell), color='C{:d}'.format(ill), linestyle='--')

    plt.show()


if __name__ == '__main__':

    from utils import data_fn, randoms_fn
    from pyrecon.utils import setup_logging

    setup_logging()
    # test_mem()
    test_dtype()
    test_iterative_fft_wrap()
    test_iterative_fft(data_fn, randoms_fn)
