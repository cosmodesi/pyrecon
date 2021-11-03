"""Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591) algorithm."""

import numpy as np

from .recon import BaseReconstruction, ReconstructionError
from . import utils


class PlaneParallelFFTReconstruction(BaseReconstruction):
    """
    Implementation of Eisenstein et al. 2007 (https://arxiv.org/pdf/astro-ph/0604362.pdf) algorithm.
    Section 3, paragraph starting with 'Restoring in full the ...'
    """
    def __init__(self, fft_engine='numpy', fft_wisdom=None, los=None, **kwargs):
        """
        Initialize :class:`IterativeFFTReconstruction`.

        Parameters
        ----------
        fft_engine : string, BaseFFTEngine, default='numpy'
            Engine for fast Fourier transforms. See :class:`BaseFFTEngine`.

        fft_wisdom : string, tuple
            Wisdom for FFTW, if ``fft_engine`` is 'fftw'.

        los : string, array, default=None
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        kwargs : dict
            See :class:`BaseReconstruction` for parameters.
        """
        super(PlaneParallelFFTReconstruction, self).__init__(los=los, **kwargs)
        kwargs = {}
        if fft_wisdom is not None: kwargs['wisdom'] = fft_wisdom
        kwargs['hermitian'] = False
        self.fft_engine = self.mesh_data.get_fft_engine(fft_engine, **kwargs)

    def set_los(self, los=None):
        """
        Set line-of-sight.

        Parameters
        ----------
        los : string, array
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.
        """
        if los is None:
            raise ReconstructionError('A (global) line-of-sight must be provided')
        if los in ['x', 'y', 'z']:
            self.los = np.zeros(3, dtype=self.mesh_data.dtype)
            self.los['xyz'.index(los)] = 1.
        else:
            los = np.array(los, dtype=self.mesh_data.dtype)
            self.los = los/utils.distance(los)

    def set_density_contrast(self, ran_min=0.75, smoothing_radius=15., **kwargs):
        r"""
        Set :math:`\delta` field :attr:`mesh_delta` from data and randoms fields :attr:`mesh_data` and :attr:`mesh_randoms`.

        Parameters
        ----------
        ran_min : float, default=0.01
            :attr:`mesh_randoms` points below this threshold times mean random weights have their density contrast set to 0.

        smoothing_radius : float, default=15
            Smoothing scale, see :meth:`RealMesh.smooth_gaussian`.

        kwargs : dict
            Optional arguments for :meth:`RealMesh.smooth_gaussian`.
        """
        if not self.has_randoms:
            self.mesh_delta = self.mesh_data/np.mean(self.mesh_data) - 1.
            self.mesh_delta /= self.bias
            self.mesh_delta.smooth_gaussian(smoothing_radius, **kwargs)
            return
        alpha = np.sum(self.mesh_data)/np.sum(self.mesh_randoms)
        self.mesh_delta = self.mesh_data - alpha*self.mesh_randoms
        mask = self.mesh_randoms > ran_min
        self.mesh_delta[mask] /= (self.bias*alpha*self.mesh_randoms[mask])
        self.mesh_delta[~mask] = 0.
        self.mesh_delta.smooth_gaussian(smoothing_radius, **kwargs)

    def run(self):
        """Run reconstruction, i.e. compute Zeldovich displacements fields :attr:`mesh_psi`."""

        delta_k = self.mesh_delta.to_complex(engine=self.fft_engine)
        k = utils.broadcast_arrays(*delta_k.coords())
        k2 = sum(k_**2 for k_ in k)
        k2[0,0,0] = 1.
        delta_k /= k2
        mu2 = sum(kk * ll for ll,kk in zip(k, self.los))/k2**0.5
        psis = []
        for iaxis in range(delta_k.ndim):
            tmp = 1j*k[iaxis]*delta_k/(1. + self.beta*mu2)
            psi = tmp.to_real(engine=self.fft_engine)
            psis.append(psi)
        self.mesh_psi = psis

    def read_shifts(self, positions, with_rsd=True):
        """
        Read Zeldovich displacement at input positions.

        Parameters
        ----------
        positions : array of shape (N,3), string
            Cartesian positions.

        with_rsd : bool, default=True
            Whether (``True``) or not (``False``) to include RSD in the shifts.
        """
        shifts = np.empty_like(positions)
        for iaxis,psi in enumerate(self.mesh_psi):
            shifts[:,iaxis] = psi.read_cic(positions)
        if with_rsd:
            shifts += self.f*np.sum(shifts*self.los,axis=-1)[:,None]*self.los
        return shifts
