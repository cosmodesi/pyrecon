"""Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591) algorithm."""

from .recon import BaseReconstruction


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
        niterations : int, default=3
            Number of iterations.
        """
        self._iter = 0
        self.mesh_delta_real = self.mesh_delta.copy()
        for iter in range(niterations):
            self._iterate()
        del self.mesh_delta
        self.mesh_psi = self._compute_psi()
        del self.mesh_delta_real

    def _iterate(self):
        if self.mpicomm.rank == 0:
            self.log_info('Running iteration {:d}.'.format(self._iter))
        # This is an implementation of eq. 22 and 24 in https://arxiv.org/pdf/1504.02591.pdf
        # \delta_{g,\mathrm{real},n} is self.mesh_delta_real
        # \delta_{g,\mathrm{red}} is self.mesh_delta
        # First compute \delta(k)/k^{2} based on current \delta_{g,\mathrm{real},n} to estimate \phi_{\mathrm{est},n} (eq. 24)
        delta_k = self.mesh_delta_real.r2c()
        for kslab, slab in zip(delta_k.slabs.x, delta_k.slabs):
            k2 = sum(kk**2 for kk in kslab)
            k2[k2 == 0.] = 1.  # avoid dividing by zero
            slab[...] /= k2

        self.mesh_delta_real = self.mesh_delta.copy()
        # Now compute \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r}
        # In the plane-parallel case (self.los is a given vector), this is simply \beta IFFT((\hat{k} \cdot \hat{\eta})^{2} \delta(k))
        if self.los is not None:
            # global los
            disp_deriv_k = delta_k.copy()
            for kslab, slab in zip(disp_deriv_k.slabs.x, disp_deriv_k.slabs):
                k2 = sum(kk**2 for kk in kslab)
                k2[k2 == 0.] = 1.  # avoid dividing by zero
                slab[...] *= sum(kk * ll for kk, ll in zip(kslab, self.los))**2  # delta_k already divided by k^{2}
            factor = self.beta
            # remove RSD part
            if self._iter == 0:
                # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                factor /= (1. + self.beta)
            self.mesh_delta_real -= factor * disp_deriv_k.c2r()
            del disp_deriv_k
        else:
            # In the local los case, \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r} is:
            # \beta \partial_{i} \partial_{j} \phi_{\mathrm{est},n} \hat{r}_{j} \hat{r}_{i}
            # i.e. \beta IFFT(k_{i} k_{j} \delta(k) / k^{2}) \hat{r}_{i} \hat{r}_{j} => 6 FFTs
            for iaxis in range(delta_k.ndim):
                for jaxis in range(iaxis, delta_k.ndim):
                    disp_deriv = delta_k.copy()
                    for kslab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs):
                        slab[...] *= kslab[iaxis] * kslab[jaxis]  # delta_k already divided by k^{2}
                    disp_deriv = disp_deriv.c2r()
                    for rslab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs):
                        rslab = self._transform_rslab(rslab)
                        r2 = sum(rr**2 for rr in rslab)
                        r2[r2 == 0.] = 1.  # avoid dividing by zero
                        slab[...] *= rslab[iaxis] * rslab[jaxis] / r2
                    factor = (1. + (iaxis != jaxis)) * self.beta  # we have j >= i and double-count j > i to account for j < i
                    if self._iter == 0:
                        # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                        factor /= (1. + self.beta)
                    # remove RSD part
                    self.mesh_delta_real -= factor * disp_deriv
        self.mesh_delta_real[...].imag = 0.
        self._iter += 1

    def _compute_psi(self):
        # Compute Zeldovich displacements given reconstructed real space density
        delta_k = self.mesh_delta_real.r2c()
        psis = []
        for iaxis in range(delta_k.ndim):
            psi = delta_k.copy()
            for kslab, slab in zip(psi.slabs.x, psi.slabs):
                k2 = sum(kk**2 for kk in kslab)
                k2[k2 == 0.] = 1.  # avoid dividing by zero
                slab[...] *= 1j * kslab[iaxis] / k2
            psis.append(psi.c2r())
            del psi
        return psis
