"""Re-implementation of Martin J. White's reconstruction code."""

import os
import ctypes

import numpy as np
from numpy import ctypeslib

from .recon import BaseReconstruction, ReconstructionError
from . import utils


class OriginalMultiGridReconstruction(BaseReconstruction):
    """
    :mod:`ctypes`-based implementation for Martin J. White's reconstruction code,
    using full multigrid V-cycle based on damped Jacobi iteration.
    We re-implemented https://github.com/martinjameswhite/recon_code/blob/master/multigrid.cpp, allowing for non-cubic (rectangular) mesh.
    Numerical agreement in the Zeldovich displacements between original code and this re-implementation is numerical precision (absolute and relative difference of 1e-10).
    To test this, change float to double and increase precision in io.cpp/write_data in the original code.
    """
    _path_lib = os.path.join(utils.lib_dir, 'multigrid_{}.so')

    @staticmethod
    def _select_nmesh(nmesh):
        # Return mesh size, equal or larger than nmesh, that can be written as 2**n, 3 * 2**n, 5 * 2**n or 7 * 2**n
        toret = []
        for n in nmesh:
            nbits = int(n).bit_length() - 1
            ntries = [2**nbits, 3 * 2**(nbits - 1), 5 * 2**(nbits - 2), 7 * 2**(nbits - 2), 2**(nbits + 1)]
            mindiff, iclosest = n, None
            for itry, ntry in enumerate(ntries):
                diff = ntry - n
                if diff >= 0 and diff < mindiff:
                    mindiff, iclosest = diff, itry
            toret.append(ntries[iclosest])
        return np.array(toret, dtype='i8')

    def __init__(self, *args, **kwargs):
        """
        Initialize :class:`MultiGridReconstruction`.
        See :class:`BaseReconstruction` for input parameters.
        """
        # Only 2 FFTs to perform, for the Gaussian smoothing, so no need to spend time on scheduling
        kwargs.setdefault('fft_plan', 'estimate')
        super(OriginalMultiGridReconstruction, self).__init__(*args, **kwargs)
        self._type_float = self.mesh_data._type_float
        self._lib = ctypes.CDLL(self._path_lib.format(self.mesh_data._precision), mode=ctypes.RTLD_LOCAL)

    def set_density_contrast(self, ran_min=0.75, smoothing_radius=15., **kwargs):
        r"""
        Set :math:`\delta` field :attr:`mesh_delta` from data and randoms fields :attr:`mesh_data` and :attr:`mesh_randoms`.

        Note
        ----
        This method follows Martin's reconstruction code: we are not satisfied with the ``ran_min`` prescription.
        At least ``ran_min`` should depend on random weights. See also Martin's notes below.

        Parameters
        ----------
        ran_min : float, default=0.75
            :attr:`mesh_randoms` points below this threshold have their density contrast set to 0.

        smoothing_radius : float, default=15
            Smoothing scale, see :meth:`RealMesh.smooth_gaussian`.

        kwargs : dict
            Optional arguments for :meth:`RealMesh.smooth_gaussian`.
        """
        if not self.has_randoms:
            self.mesh_delta = self.mesh_data / np.mean(self.mesh_data) - 1.
            self.mesh_delta /= self.bias
            self.mesh_delta.smooth_gaussian(smoothing_radius, **kwargs)
            return
        # Martin's notes:
        # We remove any points which have too few randoms for a decent
        # density estimate -- this is "fishy", but it tames some of the
        # worst swings due to 1/eps factors. Better would be an interpolation
        # or a pre-smoothing (or many more randoms).
        mask = self.mesh_randoms >= ran_min
        # alpha = np.sum(self.mesh_data[mask])/np.sum(self.mesh_randoms[mask])
        # Following two lines are how things are done in original code
        self.mesh_data[(self.mesh_randoms > 0) & (self.mesh_randoms < ran_min)] = 0.
        alpha = np.sum(self.mesh_data) / np.sum(self.mesh_randoms[mask])
        self.mesh_data[mask] /= alpha * self.mesh_randoms[mask]
        self.mesh_delta = self.mesh_data
        del self.mesh_data
        del self.mesh_randoms
        self.mesh_delta -= 1
        self.mesh_delta[~mask] = 0.
        self.mesh_delta /= self.bias
        # At this stage also remove the mean, so the source is genuinely mean 0.
        # So as to not disturb the
        # padding regions, we only compute and subtract the mean for the
        # regions with delta != 0.
        mask = self.mesh_delta != 0.
        self.mesh_delta[mask] -= np.mean(self.mesh_delta[mask])
        self.mesh_delta.smooth_gaussian(smoothing_radius, **kwargs)

    def run(self, jacobi_damping_factor=0.4, jacobi_niterations=5, vcycle_niterations=6):
        """
        Run reconstruction, i.e. set displacement potential attr:`mesh_phi` from :attr:`mesh_delta`.
        Default parameter values are the same as in Martin's code.

        Parameters
        ----------
        jacobi_damping_factor : float, default=0.4
            Damping factor for Jacobi iterations.

        jacobi_niterations : int, default=5
            Number of Jacobi iterations.

        vcycle_niterations : int, default=6
            Number of V-cycle calls.
        """
        func = self._lib.fmg
        ndim = 3
        type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_size_t, shape=ndim)
        type_boxsize = ctypeslib.ndpointer(dtype=self._type_float, shape=ndim)
        type_pointer = ctypes.POINTER(self._type_float)
        func.argtypes = (self.mesh_delta._type_float_mesh, self.mesh_delta._type_float_mesh,
                         type_nmesh, type_boxsize, type_boxsize,
                         self._type_float, self._type_float, ctypes.c_int, ctypes.c_int,
                         type_pointer if self.los is None else type_boxsize)
        self.mesh_phi = self.mesh_delta.zeros_like()
        self.mesh_phi.value.shape = -1
        if self.los is None:
            los = type_pointer()
        else:
            los = self.los.astype(self._type_float, copy=False)
        self.log_info('Computing displacement potential.')
        func(self.mesh_delta.value.ravel(order='C'), self.mesh_phi.value,
             self.mesh_delta.nmesh.astype(ctypes.c_size_t, copy=False), self.mesh_delta.boxsize.astype(self._type_float, copy=False), self.mesh_delta.boxcenter.astype(self._type_float, copy=False),
             self.beta, jacobi_damping_factor, jacobi_niterations, vcycle_niterations, los)
        del self.mesh_delta
        self.mesh_phi.value.shape = self.mesh_phi.shape

    def read_shifts(self, positions, field='disp+rsd'):
        """
        Read displacement at input positions by deriving the computed displacement potential :attr:`mesh_phi` (finite difference scheme).
        See :meth:`BaseReconstruction.read_shifts` for input parameters.
        """
        field = field.lower()
        allowed_fields = ['disp', 'rsd', 'disp+rsd']
        if field not in allowed_fields:
            raise ReconstructionError('Unknown field {}. Choices are {}'.format(field, allowed_fields))
        shifts = self.mesh_phi.read_finite_difference_cic(positions, wrap=self.wrap)
        if field == 'disp':
            return shifts
        if self.los is None:
            los = positions / utils.distance(positions)[:, None]
        else:
            los = self.los
        rsd = self.f * np.sum(shifts * los, axis=-1)[:, None] * los
        if field == 'rsd':
            return rsd
        # field == 'disp+rsd'
        shifts += rsd
        return shifts


class MultiGridReconstruction(OriginalMultiGridReconstruction):

    """Any update / test / improvement upon original algorithm."""

    def set_density_contrast(self, *args, **kwargs):
        """See :class:`BaseReconstruction.set_density_contrast`."""
        BaseReconstruction.set_density_contrast(self, *args, **kwargs)
