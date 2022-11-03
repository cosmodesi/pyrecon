import os

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
    # this is Lagrangian bias, Eulerian bias - 1
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
        print(cat['Position'].cmin(axis=0), cat['Position'].cmax(axis=0))

    data.write(data_fn)
    randoms.write(randoms_fn)


def main():

    setup_logging()
    mkdir(catalog_dir)
    save_box_lognormal_catalogs(box_data_fn, seed=42)
    #save_lognormal_catalogs(data_fn, randoms_fn, seed=42)


if __name__ == '__main__':

    main()
