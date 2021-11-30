import os
import tempfile

import numpy as np
from matplotlib import pyplot as plt

# For mockfactory installation, see https://github.com/adematti/mockfactory
from mockfactory import EulerianLinearMock, LagrangianLinearMock, utils, setup_logging
# For cosmoprimo installation see https://cosmoprimo.readthedocs.io/en/latest/user/building.html
from cosmoprimo.fiducial import DESI

from pyrecon import MultiGridReconstruction, PlaneParallelFFTReconstruction, RealMesh
from pyrecon.metrics import MeshFFTCorrelator, MeshFFTPropagator, MeshFFTTransfer, CatalogMesh


def test_metrics():
    z = 1.
    # Load DESI fiducial cosmology
    cosmo = DESI()
    power = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
    f = cosmo.sigma8_z(z=z,of='theta_cb')/cosmo.sigma8_z(z=z,of='delta_cb') # growth rate

    bias, nbar, nmesh, boxsize, boxcenter, los = 2.0, 1e-3, 128, 1000., 500., (1,0,0)
    mock = LagrangianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=42, unitary_amplitude=False)
    # This is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias-1)
    mesh_real = mock.mesh_delta_r + 1.
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=43)
    mock.set_rsd(f=f, los=los)
    data = mock.to_catalog()
    offset = data.boxcenter - data.boxsize/2.
    data['Position'] = (data['Position'] - offset) % data.boxsize + offset

    #recon = MultiGridReconstruction(f=f, bias=bias, los=los, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, fft_engine='fftw')
    recon = PlaneParallelFFTReconstruction(f=f, bias=bias, los=los, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, fft_engine='fftw')
    recon.assign_data(data.gget('Position'))
    recon.set_density_contrast()
    # Run reconstruction
    recon.run()

    from mockfactory.make_survey import RandomBoxCatalog
    randoms = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=44)

    data['Position_rec'] = data['Position'] - recon.read_shifts(data['Position'], field='disp+rsd')
    randoms['Position_rec'] = randoms['Position'] - recon.read_shifts(randoms['Position'], field='disp')
    offset = data.boxcenter - data.boxsize/2.
    for catalog in [data, randoms]:
        catalog['Position_rec'] = (catalog['Position_rec'] - offset) % catalog.boxsize + offset

    kedges = np.arange(0.005, 0.4, 0.005)
    #kedges = np.arange(0.005, 0.4, 0.05)
    muedges = np.linspace(-1., 1., 5)

    def get_correlator():
        mesh_recon = CatalogMesh(data['Position_rec'], shifted_positions=randoms['Position_rec'],
                                 boxsize=boxsize, boxcenter=boxcenter, nmesh=nmesh, resampler='cic', interlacing=2, position_type='pos')
        return MeshFFTCorrelator(mesh_recon, mesh_real, edges=(kedges, muedges), los=los)

    def get_propagator(growth=1.):
        mesh_recon = CatalogMesh(data['Position_rec'], shifted_positions=randoms['Position_rec'],
                                 boxsize=boxsize, boxcenter=boxcenter, nmesh=nmesh, resampler='cic', interlacing=2, position_type='pos')
        return MeshFFTPropagator(mesh_recon, mesh_real, edges=(kedges, muedges), los=los, growth=growth)

    def get_transfer(growth=1.):
        mesh_recon = CatalogMesh(data['Position_rec'], shifted_positions=randoms['Position_rec'],
                                 boxsize=boxsize, boxcenter=boxcenter, nmesh=nmesh, resampler='cic', interlacing=2, position_type='pos')
        return MeshFFTTransfer(mesh_recon, mesh_real, edges=(kedges, muedges), los=los, growth=growth)

    def get_propagator_ref():
        # Taken from https://github.com/cosmodesi/desi_cosmosim/blob/master/reconstruction/propagator_and_multipole/DESI_Recon/propagator_catalog_calc.py
        from nbodykit.lab import ArrayMesh, FFTPower
        meshp = data.to_nbodykit().to_mesh(position='Position_rec', Nmesh=nmesh, BoxSize=boxsize, resampler='cic', compensated=True, interlaced=True)
        meshran = randoms.to_nbodykit().to_mesh(position='Position_rec', Nmesh=nmesh, BoxSize=boxsize, resampler='cic', compensated=True, interlaced=True)
        #mesh_recon = ArrayMesh(meshp.compute() - meshran.compute(), BoxSize=boxsize)
        mesh_recon = meshp.compute() - meshran.compute()
        Nmu = len(muedges) - 1
        kmin, kmax, dk = kedges[0], kedges[-1]+1e-9, kedges[1] - kedges[0]
        r_cross = FFTPower(mesh_real, mode='2d', Nmesh=nmesh, Nmu=Nmu, dk=dk, second=mesh_recon, los=los, kmin=kmin, kmax=kmax)
        r_auto = FFTPower(mesh_recon, mode='2d', Nmesh=nmesh, Nmu=Nmu, dk=dk, los=los, kmin=kmin, kmax=kmax)
        r_auto_init = FFTPower(mesh_real, mode='2d', Nmesh=nmesh, Nmu=Nmu, dk=dk, los=los, kmin=kmin, kmax=kmax)
        return (r_cross.power['power']/r_auto_init.power['power']).real/bias

    propagator_ref = get_propagator_ref()
    correlator = get_correlator()

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'tmp.npy')
        correlator.save(fn)
        correlator = MeshFFTCorrelator.load(fn)

    propagator = correlator.to_propagator(growth=bias)
    mask = ~np.isnan(propagator_ref)
    assert np.allclose(propagator.ratio[mask], propagator_ref[mask], atol=1e-6, rtol=1e-4)
    transfer = correlator.to_transfer(growth=bias)

    assert np.allclose(get_propagator(growth=bias).ratio[mask], propagator.ratio[mask])
    assert np.allclose(get_transfer(growth=bias).ratio[mask], transfer.ratio[mask])

    fig, lax = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
    fig.subplots_adjust(wspace=0.3)
    lax = lax.flatten()
    for mu in [0., 0.7]:
        k = correlator.k(mu=mu)
        mask = k < 0.6
        k = k[mask]
        mu = np.nanmean(correlator.mu(mu=mu)[mask])
        lax[0].plot(correlator.k(mu=mu)[mask], correlator(mu=mu)[mask], label=r'$\mu = {:.2f}$'.format(mu))
        lax[1].plot(transfer.k(mu=mu)[mask], transfer(mu=mu)[mask], label=r'$\mu = {:.2f}$'.format(mu))
        lax[2].plot(propagator.k(mu=mu)[mask], propagator(mu=mu)[mask], label=r'$\mu = {:.2f}$'.format(mu))
    for ax in lax:
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('$k$ [$\mathrm{Mpc}/h$]')
    lax[0].set_ylabel(r'$r(k) = P_{\mathrm{rec},\mathrm{init}}/\sqrt{P_{\mathrm{rec}}P_{\mathrm{init}}}$')
    lax[1].set_ylabel(r'$t(k) = P_{\mathrm{rec}}/P_{\mathrm{init}}$')
    lax[2].set_ylabel(r'$g(k) = P_{\mathrm{rec},\mathrm{init}}/P_{\mathrm{init}}$')
    plt.show()

    ax = plt.gca()
    auto = correlator.auto_initial
    auto.rebin((1, len(auto.edges[-1])-1))
    ax.plot(auto.k[:,0], auto.k[:,0]*auto.power[:,0]*bias**2, label='initial')
    auto = correlator.auto_reconstructed
    auto.rebin((1, len(auto.edges)-1))
    ax.plot(auto.k[:,0], auto.k[:,0]*auto.power[:,0], label='reconstructed')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # Set up logging
    setup_logging()
    test_metrics()
