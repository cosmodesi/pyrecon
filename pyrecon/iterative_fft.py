"""Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591) algorithm."""

import numpy as np

from .recon import BaseReconstruction
from . import utils


class IterativeFFTReconstruction(BaseReconstruction):
    """
    Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591)
    field-level (as opposed to :class:`IterativeFFTParticleReconstruction`) algorithm.
    """
    def __init__(self, fft_engine='numpy', fft_wisdom=None, **kwargs):
        """
        Initialize :class:`IterativeFFTReconstruction`.

        Parameters
        ----------
        fft_engine : string, BaseFFTEngine, default='numpy'
            Engine for fast Fourier transforms. See :class:`BaseFFTEngine`.

        fft_wisdom : string, tuple
            Wisdom for FFTW, if ``fft_engine`` is 'fftw'.

        kwargs : dict
            See :class:`BaseReconstruction` for parameters.
        """
        super(IterativeFFTReconstruction,self).__init__(**kwargs)
        kwargs = {}
        if fft_wisdom is not None: kwargs['wisdom'] = fft_wisdom
        kwargs['hermitian'] = False
        self.fft_engine = self.mesh_data.get_fft_engine(fft_engine, **kwargs)

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

    def run(self, niterations=3):
        """
        Run reconstruction, i.e. compute Zeldovich displacements fields :attr:`mesh_psi`.

        Parameters
        ----------
        niterations : int
            Number of iterations.
        """
        self._iter = 0
        self.mesh_delta_real = self.mesh_delta.deepcopy()
        for iter in range(niterations):
            self._iterate()
        self.mesh_psi = self._compute_psi()

    def _iterate(self):
        self.log_info('Running iteration {:d}.'.format(self._iter))
        delta_k = self.mesh_delta_real.to_complex(engine=self.fft_engine)
        k = utils.broadcast_arrays(*delta_k.coords())
        k2 = sum(k_**2 for k_ in k)
        k2[0,0,0] = 1.
        delta_k /= k2
        self.mesh_delta_real = self.mesh_delta.deepcopy()
        if self.los is not None:
            # global los
            for i in range(delta_k.ndim):
                if self.los[i] == 0.: continue
                disp_deriv_k = (self.los[i]*k[i])**2*delta_k
                delta_rsd = disp_deriv_k.to_real(engine=self.fft_engine)
                factor = self.beta
                # remove RSD part
                if self._iter == 0:
                    # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                    factor /= (1. + self.beta)
                self.mesh_delta_real -= factor*delta_rsd
        else:
            # local los
            x = utils.broadcast_arrays(*self.mesh_delta.coords())
            x2 = sum(x_**2 for x_ in x)
            for i in range(delta_k.ndim):
                for j in range(i,delta_k.ndim):
                    disp_deriv_k = k[i]*k[j]*delta_k
                    delta_rsd = x[i]*x[j]*disp_deriv_k.to_real(engine=self.fft_engine)/x2
                    factor = (1. + (i != j)) * self.beta
                    if self._iter == 0:
                        # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                        factor /= (1. + self.beta)
                    # remove RSD part
                    self.mesh_delta_real -= factor*delta_rsd
        self._iter += 1

    def _compute_psi(self):
        # compute Zeldovich displacements given reconstructed real space density
        delta_k = self.mesh_delta_real.to_complex(engine=self.fft_engine)
        k = utils.broadcast_arrays(*delta_k.coords())
        k2 = sum(k_**2 for k_ in k)
        k2[0,0,0] = 1.
        delta_k /= k2
        delta_k[0,0,0] = 0.
        psis = []
        for iaxis in range(delta_k.ndim):
            psi = (1j*k[iaxis]*delta_k).to_real(engine=self.fft_engine)
            psis.append(psi)
        return psis

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
            if self.los is None:
                los = positions/utils.distance(positions)[:,None]
            else:
                los = self.los
            shifts += self.f*np.sum(shifts*los,axis=-1)[:,None]*los
        return shifts
