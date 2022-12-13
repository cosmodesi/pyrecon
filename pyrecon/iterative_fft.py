"""Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591) algorithm."""

from .recon import BaseReconstruction
from . import utils


class IterativeFFTReconstruction(BaseReconstruction):
    """
    Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591)
    field-level (as opposed to :class:`IterativeFFTParticleReconstruction`) algorithm.
    """
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
        del self.mesh_delta
        self.mesh_psi = self._compute_psi()
        del self.mesh_delta_real

    def _iterate(self):
        self.log_info('Running iteration {:d}.'.format(self._iter))
        # This is an implementation of eq. 22 and 24 in https://arxiv.org/pdf/1504.02591.pdf
        # \delta_{g,\mathrm{real},n} is self.mesh_delta_real
        # \delta_{g,\mathrm{red}} is self.mesh_delta
        # First compute \delta(k)/k^{2} based on current \delta_{g,\mathrm{real},n} to estimate \phi_{\mathrm{est},n} (eq. 24)
        delta_k = self.mesh_delta_real.to_complex()
        k = utils.broadcast_arrays(*delta_k.coords())
        delta_k.prod_sum([k**2 for k in delta_k.coords()], exp=-1)
        delta_k[0, 0, 0] = 0.
        self.mesh_delta_real = self.mesh_delta.deepcopy()
        # Now compute \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r}
        # In the plane-parallel case (self.los is a given vector), this is simply \beta IFFT((\hat{k} \cdot \hat{\eta})^{2} \delta(k))
        if self.los is not None:
            # global los
            disp_deriv_k = sum(kk * ll for ll, kk in zip(k, self.los))**2 * delta_k  # delta_k already divided by k^{2}
            delta_rsd = disp_deriv_k.to_real()
            factor = self.beta
            # remove RSD part
            if self._iter == 0:
                # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                factor /= (1. + self.beta)
            self.mesh_delta_real -= factor * delta_rsd
        else:
            # In the local los case, \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r} is:
            # \beta \partial_{i} \partial_{j} \phi_{\mathrm{est},n} \hat{r}_{j} \hat{r}_{i}
            # i.e. \beta IFFT(k_{i} k_{j} \delta(k) / k^{2}) \hat{r}_{i} \hat{r}_{j} => 6 FFTs
            x = utils.broadcast_arrays(*self.mesh_delta.coords())
            x2 = sum(x_**2 for x_ in x)
            for i in range(delta_k.ndim):
                for j in range(i, delta_k.ndim):
                    disp_deriv_k = k[i] * k[j] * delta_k  # delta_k already divided by k^{2}
                    delta_rsd = x[i] * x[j] * disp_deriv_k.to_real() / x2
                    factor = (1. + (i != j)) * self.beta  # we have j >= i and double-count j > i to account for j < i
                    if self._iter == 0:
                        # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                        factor /= (1. + self.beta)
                    # remove RSD part
                    self.mesh_delta_real -= factor * delta_rsd
        self._iter += 1

    def _compute_psi(self):
        # compute Zeldovich displacements given reconstructed real space density
        delta_k = self.mesh_delta_real.to_complex()
        k = utils.broadcast_arrays(*delta_k.coords())
        k2 = sum(kk**2 for kk in k)
        k2[0, 0, 0] = 1.
        delta_k /= k2
        delta_k[0, 0, 0] = 0.
        psis = []
        for iaxis in range(delta_k.ndim):
            psi = (1j * k[iaxis] * delta_k).to_real()
            psis.append(psi)
        return psis
