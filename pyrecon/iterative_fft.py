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

    def run(self, niterations=3):
        """
        Run reconstruction, i.e. compute Zeldovich displacement fields :attr:`mesh_psi`.

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
        # This is an implementation of eq. 22 and 24 in https://arxiv.org/pdf/1504.02591.pdf
        # \delta_{g,\mathrm{real},n} is self.mesh_delta_real
        # \delta_{g,\mathrm{red}} is self.mesh_delta
        # First compute \delta(k)/k^{2} based on current \delta_{g,\mathrm{real},n} to estimate \phi_{\mathrm{est},n} (eq. 24)
        delta_k = self.mesh_delta_real.to_complex(engine=self.fft_engine)
        k = utils.broadcast_arrays(*delta_k.coords())
        k2 = sum(k_**2 for k_ in k)
        k2[0,0,0] = 1. # to avoid dividing by 0
        delta_k /= k2
        self.mesh_delta_real = self.mesh_delta.deepcopy()
        # Now compute \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r}
        # In the plane-parallel case (self.los is a given vector), this is simply \beta IFFT((\hat{k} \cdot \hat{\eta})^{2} \delta(k))
        if self.los is not None:
            # global los
            for i in range(delta_k.ndim):
                if self.los[i] == 0.: continue
                disp_deriv_k = (self.los[i]*k[i])**2 * delta_k # delta_k already divided by k^{2}
                delta_rsd = disp_deriv_k.to_real(engine=self.fft_engine)
                factor = self.beta
                # remove RSD part
                if self._iter == 0:
                    # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                    factor /= (1. + self.beta)
                self.mesh_delta_real -= factor*delta_rsd
        else:
            # In the local los case, \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r} is:
            # \beta \partial_{i} \partial_{j} \phi_{\mathrm{est},n} \hat{r}_{j} \hat{r}_{i}
            # i.e. \beta IFFT(k_{i} k_{j} \delta(k) / k^{2}) \hat{r}_{i} \hat{r}_{j} => 6 FFTs
            x = utils.broadcast_arrays(*self.mesh_delta.coords())
            x2 = sum(x_**2 for x_ in x)
            for i in range(delta_k.ndim):
                for j in range(i,delta_k.ndim):
                    disp_deriv_k = k[i]*k[j]*delta_k # delta_k already divided by k^{2}
                    delta_rsd = x[i]*x[j]*disp_deriv_k.to_real(engine=self.fft_engine)/x2
                    factor = (1. + (i != j)) * self.beta # we have j >= i and double-count j > i to account for j < i
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
