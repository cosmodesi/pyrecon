import os

import numpy as np
from mockfactory import LagrangianLinearMock, RandomBoxCatalog, Catalog, cartesian_to_sky, DistanceToRedshift, setup_logging


catalog_dir = '_catalogs'
box_data_fn = os.path.join(catalog_dir, 'box_data.fits')
data_fn = os.path.join(catalog_dir, 'data.fits')
randoms_fn = os.path.join(catalog_dir, 'randoms.fits')


def catalog_rec_fn(fn, algorithm):
    base, ext = os.path.splitext(fn)
    return '{}_{}{}'.format(base, algorithm, ext)


def mkdir(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        pass


def get_random_catalog(csize=100000, boxsize=1000., seed=42):
    import mpytools as mpy
    catalog = RandomBoxCatalog(csize=csize, boxsize=boxsize, seed=seed)
    catalog['Weight'] = mpy.random.MPIRandomState(size=catalog.size, seed=seed).uniform(0.5, 1.)
    return catalog


def save_box_lognormal_catalogs(data_fn, seed=42):
    from cosmoprimo.fiducial import DESI
    z, bias, nbar, nmesh, boxsize, boxcenter = 0.7, 2.0, 3e-4, 256, 800., 0.
    cosmo = DESI()
    pklin = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
    f = cosmo.sigma8_z(z=z, of='theta_cb') / cosmo.sigma8_z(z=z, of='delta_cb')  # growth rate
    mock = LagrangianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=42, unitary_amplitude=False)
    offset = boxcenter - boxsize / 2.
    # this is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias - 1)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=43)
    mock.set_rsd(f=f, los='x')
    catalog = mock.to_catalog()
    catalog['Position'] = (catalog['Position'] - offset) % boxsize + offset
    catalog.write(box_data_fn)


def save_lognormal_catalogs(data_fn, randoms_fn, seed=42):
    from cosmoprimo.fiducial import DESI
    z, bias, nbar, nmesh, boxsize = 0.7, 2.0, 3e-4, 256, 800.
    cosmo = DESI()
    d2z = DistanceToRedshift(cosmo.comoving_radial_distance)
    pklin = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
    f = cosmo.sigma8_z(z=z, of='theta_cb') / cosmo.sigma8_z(z=z, of='delta_cb')  # growth rate
    dist = cosmo.comoving_radial_distance(z)
    boxcenter = [dist, 0, 0]
    mock = LagrangianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=42, unitary_amplitude=False)
    # This is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias - 1)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=43)
    mock.set_rsd(f=f, los=None)
    data = mock.to_catalog()

    # We've got data, now turn to randoms
    randoms = RandomBoxCatalog(nbar=10. * nbar, boxsize=boxsize, boxcenter=boxcenter, seed=44)
    # Add columns to test pyrecon script
    for cat in [data, randoms]:
        cat['Weight'] = cat.ones()
        cat['NZ'] = nbar * cat.ones()
        dist, cat['RA'], cat['DEC'] = cartesian_to_sky(cat['Position'])
        cat['Z'] = d2z(dist)

    data.write(data_fn)
    randoms.write(randoms_fn)


def test_mpi(algorithm):
    from pyrecon.utils import cartesian_to_sky
    from pyrecon import mpi
    data, randoms = get_random_catalog(seed=42), get_random_catalog(seed=81)
    gathered_data, gathered_randoms = data.gather(mpiroot=0), randoms.gather(mpiroot=0)
    mpicomm = data.mpicomm

    def get_shifts(data, randoms, position_type='pos', weight_type=True, mpicomm=None, mpiroot=None, mode='std'):
        data_positions, data_weights = data['Position'], data['Weight']
        randoms_positions, randoms_weights = randoms['Position'], randoms['Weight']
        if mpiroot is not None:
            data_positions, data_weights = mpi.gather(data_positions, mpicomm=mpicomm), mpi.gather(data_weights, mpicomm=mpicomm)
            randoms_positions, randoms_weights = mpi.gather(randoms_positions, mpicomm=mpicomm), mpi.gather(randoms_weights, mpicomm=mpicomm)
        if mpiroot is None or mpicomm.rank == mpiroot:
            if position_type == 'xyz':
                data_positions = data_positions.T
                randoms_positions = randoms_positions.T
            if position_type == 'rdd':
                data_positions = cartesian_to_sky(data_positions)
                randoms_positions = cartesian_to_sky(randoms_positions)
                data_positions = list(data_positions[1:]) + [data_positions[0]]
                randoms_positions = list(randoms_positions[1:]) + [randoms_positions[0]]
        if not weight_type:
            data_weights = randoms_weights = None

        if mode == 'std':
            recon = algorithm(positions=data_positions, randoms_positions=randoms_positions, nmesh=64, position_type=position_type, los='x', dtype='f8', mpicomm=mpicomm, mpiroot=mpiroot)
            assert recon.f is None
            recon.set_cosmo(f=0.8, bias=2.)
            recon.assign_data(data_positions, data_weights)
            recon.assign_randoms(randoms_positions, randoms_weights)
            recon.set_density_contrast()
            recon.run()
        else:
            recon = algorithm(f=0.8, bias=2., data_positions=data_positions, data_weights=data_weights,
                              randoms_positions=randoms_positions, randoms_weights=randoms_weights,
                              nmesh=64, position_type=position_type, los='x', dtype='f8', mpicomm=mpicomm, mpiroot=mpiroot)
        shifted_positions = recon.read_shifted_positions(data_positions, field='disp+rsd')
        if mpiroot is None or mpicomm.rank == mpiroot:
            assert np.array(shifted_positions).shape == np.array(data_positions).shape
        return recon.read_shifts(data_positions, field='disp+rsd')

    for weight_type in [True, False]:
        if mpicomm.rank == 0:
            shifts_ref = get_shifts(gathered_data, gathered_randoms, position_type='pos', weight_type=weight_type, mpicomm=gathered_data.mpicomm)

        for mpiroot in [None, 0]:
            for mode in ['std', 'fast']:
                for position_type in ['pos', 'rdd', 'xyz']:
                    shifts = get_shifts(data, randoms, position_type=position_type, weight_type=weight_type, mpicomm=mpicomm, mpiroot=mpiroot, mode=mode)
                    if mpiroot is None:
                        shifts = mpi.gather(shifts, mpicomm=mpicomm, mpiroot=0)
                    if mpicomm.rank == 0:
                        assert np.allclose(shifts, shifts_ref, rtol=1e-6)


def main():

    setup_logging()
    mkdir(catalog_dir)
    save_box_lognormal_catalogs(box_data_fn, seed=42)
    save_lognormal_catalogs(data_fn, randoms_fn, seed=42)


if __name__ == '__main__':

    main()
