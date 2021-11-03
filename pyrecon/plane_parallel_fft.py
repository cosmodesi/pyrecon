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
        Set line of sight.

        Parameters
        ----------
        los : string, array
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.
        """
        if los is None:
            raise ReconstructionError('A (global) line of sight must be provided')
        if los in ['x', 'y', 'z']:
            self.los = np.zeros(3, dtype=self.mesh_data.dtype)
            self.los['xyz'.index(los)] = 1.
        else:
            los = np.array(los, dtype=self.mesh_data.dtype)
            self.los = los/utils.distance(los)

    def run(self):
        """Run reconstruction, i.e. compute Zeldovich displacement fields :attr:`mesh_psi`."""

        delta_k = self.mesh_delta.to_complex(engine=self.fft_engine)
        k = utils.broadcast_arrays(*delta_k.coords())
        k2 = sum(k_**2 for k_ in k)
        k2[0,0,0] = 1. # to avoid dividing by 0
        delta_k /= k2
        mu2 = sum(kk * ll for ll,kk in zip(k, self.los))/k2**0.5
        psis = []
        for iaxis in range(delta_k.ndim):
            tmp = 1j*k[iaxis]*delta_k/(1. + self.beta*mu2)
            psi = tmp.to_real(engine=self.fft_engine)
            psis.append(psi)
        self.mesh_psi = psis
