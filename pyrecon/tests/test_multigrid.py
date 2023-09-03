import os
import time
import subprocess

import numpy as np

from pyrecon.multigrid import OriginalMultiGridReconstruction, MultiGridReconstruction
from pyrecon.utils import distance, MemoryMonitor
from pyrecon import mpi
from utils import get_random_catalog, Catalog, test_mpi


def test_mem():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=84)

    with MemoryMonitor() as mem:
        recon = MultiGridReconstruction(f=0.8, bias=2., positions=randoms['Position'], nmesh=256, dtype='f8')
        mem('init')
        recon.assign_data(data['Position'], data['Weight'])
        mem('data')
        recon.assign_randoms(randoms['Position'], randoms['Weight'])
        mem('randoms')
        recon.set_density_contrast()
        mem('delta')
        recon.run()
        mem('recon')  # 1 mesh


def test_random():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=84)
    recon = MultiGridReconstruction(f=0.8, bias=2., positions=randoms['Position'], nmesh=8, dtype='f8')
    recon.assign_data(data['Position'], data['Weight'])
    recon.assign_randoms(randoms['Position'], randoms['Weight'])
    recon.set_density_contrast()
    #recon.run(jacobi_niterations=1, vcycle_niterations=1)
    recon.run()
    # print(recon.read_shifts(data['Position']))
    # print(np.abs(recon.read_shifts(data['Position'])).max())
    # assert np.all(np.abs(recon.read_shifts(data['Position'])) < 10.)


def test_no_nrandoms():
    boxsize = 1000.
    data = get_random_catalog(boxsize=boxsize, seed=42)
    recon = MultiGridReconstruction(f=0.8, bias=2., los='x', boxcenter=0., boxsize=boxsize, nmesh=8, dtype='f8')
    recon.assign_data(data['Position'], data['Weight'])
    assert not recon.has_randoms
    recon.set_density_contrast()
    assert np.allclose(recon.mesh_delta.csum(), 0.)
    recon.run(jacobi_niterations=1, vcycle_niterations=1)
    # recon.run()
    assert np.all(np.abs(recon.read_shifts(data['Position'])) < 2.)


def test_dtype():
    # ran_min threshold in set_density_contrast() may not mask exactly the same number of cells in f4 and f8 cases, hence big difference in the end
    # With current seeds masks are the same in f4 and f8 cases
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=81)
    for los in [None, 'x']:
        all_shifts = []
        for dtype in ['f4', 'f8']:
            dtype = np.dtype(dtype)
            itemsize = np.empty(0, dtype=dtype).real.dtype.itemsize
            recon = MultiGridReconstruction(f=0.8, bias=2., positions=randoms['Position'], nmesh=64, los=los, dtype=dtype)
            recon.assign_data(data['Position'], data['Weight'])
            recon.assign_randoms(randoms['Position'], randoms['Weight'])
            recon.set_density_contrast()
            assert recon.mesh_delta.dtype.itemsize == itemsize
            recon.run()
            assert recon.mesh_phi.dtype.itemsize == itemsize
            all_shifts2 = []
            for dtype2 in ['f4', 'f8']:
                dtype2 = np.dtype(dtype2)
                shifts = recon.read_shifts(data['Position'].astype(dtype2), field='disp+rsd')
                assert shifts.dtype.itemsize == dtype2.itemsize
                all_shifts2.append(shifts)
                if dtype2 == dtype: all_shifts.append(shifts)
            assert np.allclose(*all_shifts2, atol=1e-2, rtol=1e-2)
        assert np.allclose(*all_shifts, atol=5e-2, rtol=5e-2)


def test_nmesh():
    randoms = get_random_catalog(seed=81)
    recon = MultiGridReconstruction(f=0.8, bias=2., positions=randoms['Position'], cellsize=[10, 8, 9])
    assert np.all(recon.nmesh % 2 == 0)

    import pytest
    with pytest.warns():
        recon = MultiGridReconstruction(f=0.8, bias=2., positions=randoms['Position'], nmesh=[12, 14, 18])


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
        recon = MultiGridReconstruction(f=0.8, bias=2, los='z', boxsize=boxsize, boxcenter=boxcenter, nmesh=64, wrap=True)
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


def test_los():
    boxsize = 1000.
    data = get_random_catalog(boxsize=boxsize, seed=42)
    randoms = get_random_catalog(boxsize=boxsize, seed=84)
    recon = MultiGridReconstruction(f=0.8, bias=2., los='x', boxcenter=0., boxsize=boxsize, nmesh=64, dtype='f8')
    recon.assign_data(data['Position'], data['Weight'])
    recon.assign_randoms(randoms['Position'], randoms['Weight'])
    recon.set_density_contrast()
    recon.run()
    shifts_global = recon.read_shifts(data['Position'], field='disp+rsd')
    offset = 1e8
    data['Position'][:, 0] += offset
    randoms['Position'][:, 0] += offset
    recon = MultiGridReconstruction(f=0.8, bias=2., boxcenter=[offset, 0, 0], boxsize=boxsize, nmesh=64, dtype='f8')
    recon.assign_data(data['Position'], data['Weight'])
    recon.assign_randoms(randoms['Position'], randoms['Weight'])
    recon.set_density_contrast()
    recon.run()
    shifts_local = recon.read_shifts(data['Position'], field='disp+rsd')
    assert np.allclose(shifts_local, shifts_global, rtol=1e-3, atol=1e-3)


def compute_ref(data_fn, randoms_fn, output_data_fn, output_randoms_fn):

    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    def comoving_distance(z):
        return cosmo.comoving_distance(z).value * cosmo.h

    input_fn = [fn.replace('.fits', '.rdzw') for fn in [data_fn, randoms_fn]]

    for fn, infn in zip([data_fn, randoms_fn], input_fn):
        catalog = Catalog.read(fn)
        # distance, ra, dec = cartesian_to_sky(catalog['Position'])
        # rdzw = [ra, dec, DistanceToRedshift(comoving_distance)(distance)] + [catalog['Weight']]
        rdzw = list(catalog['Position'].T) + [catalog['Weight']]
        np.savetxt(infn, np.array(rdzw).T)

    catalog_dir = os.path.dirname(infn)
    command = '{0} {1} {2} {2} 2 0.81 15'.format(recon_code, *[os.path.basename(infn) for infn in input_fn])
    t0 = time.time()
    print(command)
    subprocess.call(command, shell=True, cwd=catalog_dir)
    print('recon_code completed in {:.2f} s'.format(time.time() - t0), flush=True)

    output_fn = [os.path.join(catalog_dir, base) for base in ['data_rec.xyzw', 'rand_rec.xyzw']]
    for infn, fn, outfn in zip([data_fn, randoms_fn], [output_data_fn, output_randoms_fn], output_fn):
        x, y, z, w = np.loadtxt(outfn, unpack=True)
        positions = np.array([x, y, z]).T
        catalog = Catalog.read(infn).gather()
        if catalog is not None:
            # print(np.mean(distance(positions - catalog['Position'])))
            catalog['Position_rec'] = positions
            catalog['Weight'] = w
            catalog.write(fn)


def test_recon(data_fn, randoms_fn, output_data_fn, output_randoms_fn):
    # boxsize = 1199.9995117188 in float32
    # boxcenter = [1753.8884277344, 400.0001831055, 400.0003662109] in float64
    boxsize = 1199.9988620158
    boxcenter = [1741.8557233434, -0.0002247471, 0.0001600799]
    recon = OriginalMultiGridReconstruction(boxsize=boxsize, boxcenter=boxcenter, nmesh=128, dtype='f8')
    mpicomm = recon.mpicomm
    recon.set_cosmo(f=0.81, bias=2.)

    # recon = OriginalMultiGridReconstruction(positions=fitsio.read(randoms_fn, columns=['Position'])['Position'], nmesh=128, dtype='f4')
    # recon.set_cosmo(f=0.81, bias=2.)
    # print(recon.mesh_data.boxsize, recon.mesh_data.boxcenter)

    nslabs = 1
    for fn, assign in zip([data_fn, randoms_fn], [recon.assign_data, recon.assign_randoms]):
        for islab in range(nslabs):
            data = Catalog.read(fn)
            start = islab * data.csize // nslabs
            stop = (islab + 1) * data.csize // nslabs
            data = data.cslice(start, stop)
            assign(data['Position'], data['Weight'])
    recon.set_density_contrast()
    # print(np.max(recon.mesh_delta))
    t0 = time.time()
    recon.run()
    #recon.run(jacobi_niterations=5, vcycle_niterations=1)
    if mpicomm.rank == 0:
        print('pyrecon completed in {:.4f} s'.format(time.time() - t0))
    # print(np.std(recon.mesh_phi))
    # recon.f = recon.beta

    for input_fn, output_fn in zip([data_fn, randoms_fn], [output_data_fn, output_randoms_fn]):
        catalog = Catalog.read(input_fn)
        shifts = recon.read_shifts(catalog['Position'], field='disp+rsd')
        catalog['Position_rec'] = catalog['Position'] - shifts
        catalog.write(output_fn)
        shifts = mpi.gather(shifts, mpicomm=mpicomm, mpiroot=0)
        if mpicomm.rank == 0:
            print('RMS', (np.mean(np.sum(shifts**2, axis=-1)) / 3)**0.5)


def compare_ref(data_fn, output_data_fn, ref_output_data_fn):
    positions = Catalog.read(data_fn)['Position']
    output_positions = Catalog.read(output_data_fn)['Position_rec']
    ref_output_positions = Catalog.read(ref_output_data_fn)['Position_rec']

    print('abs test - ref', np.max(distance(output_positions - ref_output_positions)))
    print('rel test - ref', np.max(distance(output_positions - ref_output_positions) / distance(ref_output_positions - positions)))
    print('mean diff test', np.mean(distance(output_positions - positions)))
    print('mean diff ref', np.mean(distance(ref_output_positions - positions)))
    assert np.allclose(output_positions, ref_output_positions, rtol=1e-7, atol=1e-7)


def test_script(data_fn, randoms_fn, output_data_fn, output_randoms_fn):

    catalog_dir = '_catalogs'
    command = 'pyrecon config_multigrid.yaml --data-fn {} --randoms-fn {} --output-data-fn {} --output-randoms-fn {}'.format(
              os.path.relpath(data_fn, catalog_dir), os.path.relpath(randoms_fn, catalog_dir),
              os.path.relpath(output_data_fn, catalog_dir), os.path.relpath(output_randoms_fn, catalog_dir))
    subprocess.call(command, shell=True)
    data = Catalog.read(data_fn)
    randoms = Catalog.read(randoms_fn)
    recon = MultiGridReconstruction(positions=randoms['Position'], nmesh=128, dtype='f8')
    recon.set_cosmo(f=0.8, bias=2.)
    recon.assign_data(data['Position'], data['Weight'])
    recon.assign_randoms(randoms['Position'], randoms['Weight'])
    recon.set_density_contrast()
    recon.run()

    ref_positions_rec_data = recon.read_shifted_positions(data['Position'])
    ref_positions_rec_randoms = recon.read_shifted_positions(randoms['Position'])

    data = Catalog.read(output_data_fn)
    randoms = Catalog.read(output_randoms_fn)

    # print(ref_positions_rec_data, data['Position_rec'], ref_positions_rec_data-data['Position_rec'])
    assert np.allclose(ref_positions_rec_data, data['Position_rec'])
    assert np.allclose(ref_positions_rec_randoms, randoms['Position_rec'])


def test_script_no_randoms(data_fn, output_data_fn):

    catalog_dir = '_catalogs'
    command = 'pyrecon config_multigrid_no_randoms.yaml --data-fn {} --output-data-fn {}'.format(
              os.path.relpath(data_fn, catalog_dir), os.path.relpath(output_data_fn, catalog_dir))
    subprocess.call(command, shell=True)
    data = Catalog.read(data_fn)
    boxsize = 800
    boxcenter = 0.
    recon = MultiGridReconstruction(nthreads=4, los=0, boxcenter=boxcenter, boxsize=boxsize, nmesh=128, dtype='f8')
    recon.set_cosmo(f=0.8, bias=2.)
    recon.assign_data(data['Position'])
    recon.set_density_contrast()
    recon.run()

    ref_positions_rec_data = data['Position'] - recon.read_shifts(data['Position'])
    data = Catalog.read(output_data_fn)

    # print(ref_positions_rec_data, data['Position_rec'], ref_positions_rec_data-data['Position_rec'])
    assert np.allclose(ref_positions_rec_data, data['Position_rec'])


def test_ref(data_fn, randoms_fn, data_fn_rec=None, randoms_fn_rec=None):
    boxsize = 1200.
    boxcenter = [1754, 0, 0]
    data = Catalog.read(data_fn)
    randoms = Catalog.read(randoms_fn)
    recon = MultiGridReconstruction(f=0.8, bias=2., los=None, boxcenter=boxcenter, boxsize=boxsize, nmesh=128, dtype='f8')
    recon.assign_data(data['Position'], data['Weight'])
    recon.assign_randoms(randoms['Position'], randoms['Weight'])
    recon.set_density_contrast()
    recon.run()

    from pypower import CatalogFFTPower
    from matplotlib import pyplot as plt

    for cat, fn in zip([data, randoms], [data_fn_rec, randoms_fn_rec]):
        rec = recon.read_shifted_positions(cat['Position'])
        if 'Position_rec' in cat:
            if recon.mpicomm.rank == 0: print('Checking...')
            assert np.allclose(rec, cat['Position_rec'], rtol=1e-4, atol=1e-4)
        else:
            cat['Position_rec'] = rec
        if fn is not None:
            cat.write(fn)
    exit()
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


def test_finite_difference():
    from pmesh.pm import ParticleMesh
    from pyrecon import mpi
    mpicomm = mpi.COMM_WORLD
    nmesh = 16
    pm = ParticleMesh(BoxSize=[1.] * 3, Nmesh=[nmesh] * 3, np=(mpicomm.size, 1), dtype='f8')
    mesh = pm.create('real', value=2.)
    from pyrecon import _multigrid
    size = 10
    positions = np.column_stack([np.linspace(0., 1., size)] * 3)
    boxcenter = np.array([0.5] * 3)
    values = _multigrid.read_finite_difference_cic(mesh, positions, boxcenter)
    print(values)
    print(np.abs(values).min(), np.abs(values).max())
    #if mpicomm.rank == 0: print(values)


if __name__ == '__main__':

    from utils import box_data_fn, data_fn, randoms_fn, catalog_dir, catalog_rec_fn
    from pyrecon.utils import setup_logging

    setup_logging()
    # Run utils.py to generate catalogs needed for these tests

    recon_code = os.path.join(os.path.abspath(os.path.dirname(__file__)), '_codes', 'recon')
    output_data_fn = os.path.join(catalog_dir, 'data_rec.fits')
    output_randoms_fn = os.path.join(catalog_dir, 'randoms_rec.fits')
    ref_output_data_fn = os.path.join(catalog_dir, 'ref_data_rec.fits')
    ref_output_randoms_fn = os.path.join(catalog_dir, 'ref_randoms_rec.fits')
    script_output_box_data_fn = os.path.join(catalog_dir, 'script_box_data_rec.fits')
    script_output_data_fn = os.path.join(catalog_dir, 'script_data_rec.fits')
    script_output_randoms_fn = os.path.join(catalog_dir, 'script_randoms_rec.fits')

    # test_mem()
    test_dtype()
    test_nmesh()
    test_wrap()
    test_mpi(MultiGridReconstruction)
    test_random()
    test_no_nrandoms()
    test_los()
    # test_finite_difference()
    #test_recon(data_fn, randoms_fn, output_data_fn, output_randoms_fn)
    #compute_ref(data_fn, randoms_fn, ref_output_data_fn, ref_output_randoms_fn)
    #compare_ref(data_fn, output_data_fn, ref_output_data_fn)
    #compare_ref(randoms_fn, output_randoms_fn, ref_output_randoms_fn)

    #test_script(data_fn, randoms_fn, script_output_data_fn, script_output_randoms_fn)
    #test_script_no_randoms(box_data_fn, script_output_box_data_fn)
    # compute_power_no_randoms([script_output_box_data_fn]*2, ['RSDPosition', 'Position_rec'])
    # compute_power((data_fn, randoms_fn), (output_data_fn, output_randoms_fn))
    # compute_power((data_fn, randoms_fn), (ref_output_data_fn, ref_output_randoms_fn))
    # compute_power((ref_output_data_fn, ref_output_randoms_fn), (output_data_fn, output_randoms_fn))
    data_fn_rec, randoms_fn_rec = [catalog_rec_fn(fn, 'multigrid') for fn in [data_fn, randoms_fn]]
    # test_ref(data_fn, randoms_fn, data_fn_rec, randoms_fn_rec)
    test_ref(data_fn_rec, randoms_fn_rec, None, None)
