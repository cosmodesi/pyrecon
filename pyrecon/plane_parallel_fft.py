"""Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591) algorithm."""


from .recon import BaseReconstruction, ReconstructionError


class PlaneParallelFFTReconstruction(BaseReconstruction):
    """
    Implementation of Eisenstein et al. 2007 (https://arxiv.org/pdf/astro-ph/0604362.pdf) algorithm.
    Section 3, paragraph starting with 'Restoring in full the ...'
    """
    _compressed = True

    def __init__(self, los=None, **kwargs):
        """
        Initialize :class:`IterativeFFTReconstruction`.

        Parameters
        ----------
        los : string, array, default=None
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        kwargs : dict
            See :class:`BaseReconstruction` for parameters.
        """
        super(PlaneParallelFFTReconstruction, self).__init__(los=los, **kwargs)

    def set_los(self, los=None):
        """
        Set line-of-sight.

        Parameters
        ----------
        los : string, array_like
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.
        """
        super(PlaneParallelFFTReconstruction, self).set_los(los=los)
        if self.los is None:
            raise ReconstructionError('A (global) line-of-sight must be provided')

    def run(self):
        """Run reconstruction, i.e. compute Zeldovich displacement fields :attr:`mesh_psi`."""
        delta_k = self.mesh_delta.r2c()
        del self.mesh_delta
        psis = []
        for iaxis in range(delta_k.ndim):
            psi = delta_k.copy()
            for kslab, islab, slab in zip(psi.slabs.x, psi.slabs.i, psi.slabs):
                k2 = sum(kk**2 for kk in kslab)
                k2[k2 == 0.] = 1.  # avoid dividing by zero
                mu2 = sum(kk * ll for kk, ll in zip(kslab, self.los))**2 / k2
                # i = N / 2 is pure complex, we can remove it safely
                # ... and we have to, because it is turned to real when hermitian symmetry is assumed?
                mask = islab[iaxis] != self.nmesh[iaxis] // 2
                slab[...] *= 1j * kslab[iaxis] / k2 / (1. + self.beta * mu2) * mask
            psis.append(psi.c2r())
            del psi
        self.mesh_psi = psis
