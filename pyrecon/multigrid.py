"""Re-implementation of Martin J. White's reconstruction code."""

import numpy as np

from .recon import BaseReconstruction, ReconstructionError, format_positions_wrapper
from . import _multigrid, utils, mpi


class OriginalMultiGridReconstruction(BaseReconstruction):
    """
    :mod:`ctypes`-based implementation for Martin J. White's reconstruction code,
    using full multigrid V-cycle based on damped Jacobi iteration.
    We re-implemented https://github.com/martinjameswhite/recon_code/blob/master/multigrid.cpp, allowing for non-cubic (rectangular) mesh.
    Numerical agreement in the Zeldovich displacements between original code and this re-implementation is numerical precision (absolute and relative difference of 1e-10).
    To test this, change float to double and increase precision in io.cpp/write_data in the original code.
    """
    _compressed = True

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

    def __init__(self, *args, mpicomm=mpi.COMM_WORLD, **kwargs):
        # We require a split, along axis x.
        super(OriginalMultiGridReconstruction, self).__init__(*args, decomposition=(mpicomm.size, 1), mpicomm=mpicomm, **kwargs)

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
        self.smoothing_radius = smoothing_radius
        if self.has_randoms:
            # Martin's notes:
            # We remove any points which have too few randoms for a decent
            # density estimate -- this is "fishy", but it tames some of the
            # worst swings due to 1/eps factors. Better would be an interpolation
            # or a pre-smoothing (or many more randoms).
            # alpha = np.sum(self.mesh_data[mask])/np.sum(self.mesh_randoms[mask])
            # Following two lines are how things are done in original code
            for data, randoms in zip(self.mesh_data.slabs, self.mesh_randoms.slabs):
                data[(randoms > 0) & (randoms < ran_min)] = 0.
            alpha = self.mesh_data.csum() / self.mpicomm.allreduce(sum(np.sum(randoms[randoms >= ran_min]) for randoms in self.mesh_randoms))
            for data, randoms in zip(self.mesh_data.slabs, self.mesh_randoms.slabs):
                mask = randoms >= ran_min
                data[mask] /= alpha * randoms[mask]
                data[...] -= 1.
                data[~mask] = 0.
                data[...] /= self.bias
            self.mesh_delta = self.mesh_data
            del self.mesh_data
            del self.mesh_randoms
            # At this stage also remove the mean, so the source is genuinely mean 0.
            # So as to not disturb the padding regions, we only compute and subtract the mean for the regions with delta != 0.
            mean = self.mesh_delta.csum() / self.mpicomm.allreduce(sum(np.sum(delta != 0.) for delta in self.mesh_delta))
            for delta in self.mesh_delta.slabs:
                mask = delta != 0.
                delta[mask] -= mean
        else:
            self.mesh_delta /= (self.mesh_data.cmean() * self.bias)
            self.mesh_delta -= 1. / self.bias
            del self.mesh_data
        self.mesh_delta = self._smooth_gaussian(self.mesh_delta)

    def _vcycle(self, v, f):
        _multigrid.jacobi(v, f, self.boxcenter, self.beta, damping_factor=self.jacobi_damping_factor, niterations=self.jacobi_niterations, los=self.los)
        nmesh = v.pm.Nmesh
        recurse = np.all((nmesh > 4) & (nmesh % 2 == 0))
        if recurse:
            f2h = _multigrid.reduce(_multigrid.residual(v, f, self.boxcenter, self.beta, los=self.los))
            v2h = f2h.pm.create(type='real', value=0.)
            self._vcycle(v2h, f2h)
            v.value += _multigrid.prolong(v2h).value
        _multigrid.jacobi(v, f, self.boxcenter, self.beta, damping_factor=self.jacobi_damping_factor, niterations=self.jacobi_niterations, los=self.los)

    def _fmg(self, f1h):
        nmesh = f1h.pm.Nmesh
        recurse = np.all((nmesh > 4) & (nmesh % 2 == 0))
        if recurse:
            # Recurse to a coarser grid
            v1h = _multigrid.prolong(self._fmg(_multigrid.reduce(f1h)))
        else:
            # Start with a guess of zeros
            v1h = f1h.pm.create(type='real', value=0.)
        for iter in range(self.vcycle_niterations):
            self._vcycle(v1h, f1h)
        return v1h

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
        self.jacobi_damping_factor = float(jacobi_damping_factor)
        self.jacobi_niterations = int(jacobi_niterations)
        self.vcycle_niterations = int(vcycle_niterations)
        if self.mpicomm.rank == 0: self.log_info('Computing displacement potential.')
        self.mesh_phi = self._fmg(self.mesh_delta)
        del self.mesh_delta

    @format_positions_wrapper(return_input_type=False)
    def read_shifts(self, positions, field='disp+rsd'):
        """
        Read displacement at input positions by deriving the computed displacement potential :attr:`mesh_phi` (finite difference scheme).
        See :meth:`BaseReconstruction.read_shifts` for input parameters.
        """
        field = field.lower()
        allowed_fields = ['disp', 'rsd', 'disp+rsd']
        if field not in allowed_fields:
            raise ReconstructionError('Unknown field {}. Choices are {}'.format(field, allowed_fields))
        shifts = _multigrid.read_finite_difference_cic(self.mesh_phi, positions, self.boxcenter)
        if field == 'disp':
            return shifts
        if self.los is None:
            los = positions / utils.distance(positions)[:, None]
        else:
            los = self.los.astype(shifts.dtype)
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
