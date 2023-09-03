"""Re-implementation of Bautista et al. 2018 (https://arxiv.org/pdf/1712.08064.pdf) algorithm."""

import numpy as np

from .recon import BaseReconstruction, ReconstructionError, format_positions_wrapper, format_positions_weights_wrapper
from . import utils


class OriginalIterativeFFTParticleReconstruction(BaseReconstruction):
    """
    Exact re-implementation of Bautista et al. 2018 (https://arxiv.org/pdf/1712.08064.pdf) algorithm
    at https://github.com/julianbautista/eboss_clustering/blob/master/python/recon.py.
    Numerical agreement in the Zeldovich displacements between original codes and this re-implementation is machine precision
    (absolute and relative difference of 1e-12).
    """
    _compressed = True

    @format_positions_weights_wrapper
    def assign_data(self, positions, weights=None):
        """
        Assign (paint) data to :attr:`mesh_data`.
        Keeps track of input positions (for :meth:`run`) and weights (for :meth:`set_density_contrast`).
        See :meth:`BaseReconstruction.assign_data` for parameters.
        """
        if weights is None:
            weights = np.ones_like(positions, shape=(len(positions),))
        if getattr(self, 'mesh_data', None) is None:
            self.mesh_data = self.pm.create(type='real', value=0.)
            self._positions_data = positions
            self._weights_data = weights
        else:
            self._positions_data = np.concatenate([self._positions_data, positions], axis=0)
            self._weights_data = np.concatenate([self._weights_data, weights], axis=0)
        self._paint(positions, weights=weights, out=self.mesh_data)

    def set_density_contrast(self, ran_min=0.01, smoothing_radius=15., check=False):
        r"""
        Set :math:`\delta` field :attr:`mesh_delta` from data and randoms fields :attr:`mesh_data` and :attr:`mesh_randoms`.

        Note
        ----
        This method follows Julian's reconstruction code.
        :attr:`mesh_data` and :attr:`mesh_randoms` fields are assumed to be smoothed already.

        Parameters
        ----------
        ran_min : float, default=0.01
            :attr:`mesh_randoms` points below this threshold times mean random weights have their density contrast set to 0.

        smoothing_radius : float, default=15
            Smoothing scale, see :meth:`RealMesh.smooth_gaussian`.

        check : bool, default=False
            If ``True``, run some tests (printed in logger) to assess whether enough randoms have been used.
        """
        self.ran_min = ran_min
        self.smoothing_radius = smoothing_radius

        self.mesh_delta = self.mesh_data.copy()

        if self.has_randoms:

            if check:
                nnonzero = self.mpicomm.allreduce(sum(np.sum(randoms > 0.) for randoms in self.mesh_randoms))
                if nnonzero < 2: raise ValueError('Very few randoms!')

            sum_data, sum_randoms = self.mesh_data.csum(), self.mesh_randoms.csum()
            alpha = sum_data * 1. / sum_randoms

            for delta, randoms in zip(self.mesh_delta.slabs, self.mesh_randoms.slabs):
                delta[...] -= alpha * randoms

            threshold = ran_min * sum_randoms / self._size_randoms

            for delta, randoms in zip(self.mesh_delta.slabs, self.mesh_randoms.slabs):
                mask = randoms > threshold
                delta[mask] /= (self.bias * alpha * randoms[mask])
                delta[~mask] = 0.

            if check:
                mean_nran_per_cell = self.mpicomm.allreduce(sum(randoms[randoms > 0] for randoms in self.mesh_randoms))
                std_nran_per_cell = self.mpicomm.allreduce(sum(randoms[randoms > 0]**2 for randoms in self.mesh_randoms)) - mean_nran_per_cell**2
                if self.mpicomm.rank == 0:
                    self.log_info('Mean smoothed random density in non-empty cells is {:.4f} (std = {:.4f}), threshold is (ran_min * mean weight) = {:.4f}.'.format(mean_nran_per_cell, std_nran_per_cell, threshold))

                frac_nonzero_masked = 1. - self.mpicomm.allreduce(sum(np.sum(randoms > 0.) for randoms in self.mesh_randoms)) / nnonzero
                del mask_nonzero
                if self.mpicomm.rank == 0:
                    if frac_nonzero_masked > 0.1:
                        self.log_warning('Masking a large fraction {:.4f} of non-empty cells. You should probably increase the number of randoms.'.format(frac_nonzero_masked))
                    else:
                        self.log_info('Masking a fraction {:.4f} of non-empty cells.'.format(frac_nonzero_masked))

        else:
            self.mesh_delta /= (self.mesh_delta.cmean() * self.bias)
            self.mesh_delta -= 1. / self.bias

    def run(self, niterations=3):
        """
        Run reconstruction, i.e. compute reconstructed data real-space positions (:attr:`_positions_rec_data`)
        and Zeldovich displacements fields :attr:`mesh_psi`.

        Parameters
        ----------
        niterations : int
            Number of iterations.
        """
        self._iter = 0
        # Gaussian smoothing before density contrast calculation
        self.mesh_data = self._smooth_gaussian(self.mesh_data)
        if self.has_randoms:
            self.mesh_randoms = self._smooth_gaussian(self.mesh_randoms)
        self._positions_rec_data = self._positions_data.copy()
        for iter in range(niterations):
            self.mesh_psi = self._iterate(return_psi=iter == niterations - 1)
        del self.mesh_data
        if self.has_randoms:
            del self.mesh_randoms

    def _iterate(self, return_psi=False):
        if self.mpicomm.rank == 0:
            self.log_info('Running iteration {:d}.'.format(self._iter))

        if self._iter > 0:
            self.mesh_data[...] = 0.  # to reset mesh values
            # Painting reconstructed data real-space positions
            # super in order not to save positions_rec_data
            super(OriginalIterativeFFTParticleReconstruction, self).assign_data(self._positions_rec_data, weights=self._weights_data, position_type='pos', mpiroot=None)
            # Gaussian smoothing before density contrast calculation
            self.mesh_data = self._smooth_gaussian(self.mesh_data)

        self.set_density_contrast(ran_min=self.ran_min, smoothing_radius=self.smoothing_radius)
        delta_k = self.mesh_delta.r2c()
        del self.mesh_delta

        for kslab, slab in zip(delta_k.slabs.x, delta_k.slabs):
            k2 = sum(kk**2 for kk in kslab)
            k2[k2 == 0.] = 1.  # avoid dividing by zero
            slab[...] /= k2

        if self.mpicomm.rank == 0:
            self.log_info('Computing displacement field.')

        shifts = np.empty_like(self._positions_rec_data)
        psis = []
        for iaxis in range(delta_k.ndim):
            # No need to compute psi on axis where los is 0
            if not return_psi and self.los is not None and self.los[iaxis] == 0:
                shifts[:, iaxis] = 0.
                continue

            psi = delta_k.copy()
            for kslab, islab, slab in zip(psi.slabs.x, psi.slabs.i, psi.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2
                slab[...] *= 1j * kslab[iaxis] * mask

            psi = psi.c2r()
            # Reading shifts at reconstructed data real-space positions
            shifts[:, iaxis] = self._readout(psi, self._positions_rec_data)
            if return_psi: psis.append(psi)
            del psi
        # self.log_info('A few displacements values:')
        # for s in shifts[:3]: self.log_info('{}'.format(s))
        if self.los is None:
            los = self._positions_data / utils.distance(self._positions_data)[:, None]
        else:
            los = self.los
        # Comments in Julian's code:
        # For first loop need to approximately remove RSD component from psi to speed up convergence
        # See Burden et al. 2015: 1504.02591v2, eq. 12 (flat sky approximation)
        if self._iter == 0:
            shifts -= self.beta / (1 + self.beta) * np.sum(shifts * los, axis=-1)[:, None] * los
        # Comments in Julian's code:
        # Remove RSD from original positions of galaxies to give new positions
        # these positions are then used in next determination of psi,
        # assumed to not have RSD.
        # The iterative procedure then uses the new positions as if they'd been read in from the start
        self._positions_rec_data = self._positions_data - self.f * np.sum(shifts * los, axis=-1)[:, None] * los
        self._iter += 1
        if return_psi:
            return psis

    @format_positions_wrapper(return_input_type=False)
    def read_shifts(self, positions, field='disp+rsd'):
        """
        Read displacement at input positions.

        Note
        ----
        Data shifts are read at the reconstructed real-space positions,
        while random shifts are read at the redshift-space positions, is that consistent?

        Parameters
        ----------
        positions : array of shape (N, 3), string
            Cartesian positions.
            Pass string 'data' to get the displacements for the input data positions passed to :meth:`assign_data`.
            Note that in this case, shifts are read at the reconstructed data real-space positions.

        field : string, default='disp+rsd'
            Either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        Returns
        -------
        shifts : array of shape (N, 3)
            Displacements.
        """
        field = field.lower()
        allowed_fields = ['disp', 'rsd', 'disp+rsd']
        if field not in allowed_fields:
            raise ReconstructionError('Unknown field {}. Choices are {}'.format(field, allowed_fields))

        def _read_shifts(positions):
            shifts = np.empty_like(positions)
            for iaxis, psi in enumerate(self.mesh_psi):
                shifts[:, iaxis] = self._readout(psi, positions)
            return shifts

        if isinstance(positions, str) and positions == 'data':
            # _positions_rec_data already wrapped during iteration
            shifts = _read_shifts(self._positions_rec_data)
            if field == 'disp':
                return shifts
            rsd = self._positions_data - self._positions_rec_data
            if field == 'rsd':
                return rsd
            # field == 'disp+rsd'
            shifts += rsd
            return shifts

        if self.wrap: positions = self._wrap(positions)  # wrap here for local los
        shifts = _read_shifts(positions)  # aleady wrapped

        if field == 'disp':
            return shifts

        if self.los is None:
            los = positions / utils.distance(positions)[:, None]
        else:
            los = self.los.astype(positions.dtype)
        rsd = self.f * np.sum(shifts * los, axis=-1)[:, None] * los

        if field == 'rsd':
            return rsd

        # field == 'disp+rsd'
        # we follow convention of original algorithm: remove RSD first,
        # then remove Zeldovich displacement
        real_positions = positions - rsd
        diff = real_positions - self.offset
        if (not self.wrap) and any(self.mpicomm.allgather(np.any((diff < 0) | (diff > self.boxsize - self.cellsize)))):
            if self.mpicomm.rank == 0:
                self.log_warning('Some particles are out-of-bounds.')
        shifts = _read_shifts(real_positions)

        return shifts + rsd

    @format_positions_wrapper(return_input_type=True)
    def read_shifted_positions(self, positions, field='disp+rsd'):
        """
        Read shifted positions i.e. the difference ``positions - self.read_shifts(positions, field=field)``.
        Output (and input) positions are wrapped if :attr:`wrap`.

        Parameters
        ----------
        positions : array of shape (N, 3), string
            Cartesian positions.
            Pass string 'data' to get the shift positions for the input data positions passed to :meth:`assign_data`.
            Note that in this case, shifts are read at the reconstructed data real-space positions.

        field : string, default='disp+rsd'
            Apply either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        Returns
        -------
        positions : array of shape (N, 3)
            Shifted positions.
        """
        shifts = self.read_shifts(positions, field=field, position_type='pos', mpiroot=None)
        if isinstance(positions, str) and positions == 'data':
            positions = self._positions_data
        positions = positions - shifts
        if self.wrap: positions = self._wrap(positions)
        return positions


class IterativeFFTParticleReconstruction(OriginalIterativeFFTParticleReconstruction):

    """Any update / test / improvement upon original algorithm."""
