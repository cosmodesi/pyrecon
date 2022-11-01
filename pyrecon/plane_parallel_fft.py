"""Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591) algorithm."""

from .recon import BaseReconstruction, ReconstructionError
from . import utils


class PlaneParallelFFTReconstruction(BaseReconstruction):
    """
    Implementation of Eisenstein et al. 2007 (https://arxiv.org/pdf/astro-ph/0604362.pdf) algorithm.
    Section 3, paragraph starting with 'Restoring in full the ...'
    """
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
        delta_k = self.mesh_delta.to_complex()
        del self.mesh_delta
        k = utils.broadcast_arrays(*delta_k.coords())
        k2 = sum(kk**2 for kk in k)
        k2[0, 0, 0] = 1.  # to avoid dividing by 0
        delta_k /= k2
        mu2 = sum(kk * ll for ll, kk in zip(k, self.los))**2 / k2
        psis = []
        for iaxis in range(delta_k.ndim):
            tmp = 1j * k[iaxis] * delta_k / (1. + self.beta * mu2)
            psi = tmp.to_real()
            psis.append(psi)
        self.mesh_psi = psis
