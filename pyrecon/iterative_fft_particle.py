"""Re-implementation of Bautista et al. 2018 (https://arxiv.org/pdf/1712.08064.pdf) algorithm."""

import numpy as np

from .recon import BaseReconstruction, ReconstructionError
from . import utils


class OriginalIterativeFFTParticleReconstruction(BaseReconstruction):
    """
    Exact re-implementation of Bautista et al. 2018 (https://arxiv.org/pdf/1712.08064.pdf) algorithm
    at https://github.com/julianbautista/eboss_clustering/blob/master/python/recon.py.
    Numerical agreement in the Zeldovich displacements between original codes and this re-implementation is machine precision
    (absolute and relative difference of 1e-12).
    """
    def assign_data(self, positions, weights=None):
        """
        Assign (paint) data to :attr:`mesh_data`.
        Keeps track of input positions (for :meth:`run`) and weights (for :meth:`set_density_contrast`).
        See :meth:`BaseReconstruction.assign_data` for parameters.
        """
        if weights is None:
            weights = np.ones_like(positions, shape=(len(positions),))
        if self.wrap: positions = self.info.wrap(positions)
        if self.mesh_data.value is None:
            self._positions_data = positions
            self._weights_data = weights
        else:
            self._positions_data = np.concatenate([self._positions_data, positions], axis=0)
            self._weights_data = np.concatenate([self._weights_data, weights], axis=0)
        self.mesh_data.assign_cic(positions, weights=weights, wrap=self.wrap)

    def set_density_contrast(self, ran_min=0.01, smoothing_radius=15.):
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
        """
        self.ran_min = ran_min
        self.smoothing_radius = smoothing_radius
        if self.has_randoms:
            sum_data, sum_randoms = np.sum(self.mesh_data.value), np.sum(self.mesh_randoms.value)
            alpha = sum_data * 1. / sum_randoms
            self.mesh_delta = self.mesh_data - alpha * self.mesh_randoms
            threshold = ran_min * sum_randoms / self._size_randoms
            mask = self.mesh_randoms > threshold
            self.mesh_delta[mask] /= (self.bias * alpha * self.mesh_randoms[mask])
            self.mesh_delta[~mask] = 0.
        else:
            self.mesh_delta = self.mesh_data / np.mean(self.mesh_data) - 1.
            self.mesh_delta /= self.bias

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
        self.mesh_data.smooth_gaussian(self.smoothing_radius, method='fft')
        if self.has_randoms: self.mesh_randoms.smooth_gaussian(self.smoothing_radius, method='fft')
        self._positions_rec_data = self._positions_data.copy()
        for iter in range(niterations):
            self.mesh_psi = self._iterate(return_psi=iter == niterations - 1)
        del self.mesh_randoms

    def _iterate(self, return_psi=False):
        self.log_info('Running iteration {:d}.'.format(self._iter))

        if self._iter > 0:
            self.mesh_data = self.mesh_randoms.copy()
            self.mesh_data.value = None  # to reset mesh values
            # Painting reconstructed data real-space positions
            wrap = self.wrap; self.wrap = True  # enforce wrapping
            super(OriginalIterativeFFTParticleReconstruction, self).assign_data(self._positions_rec_data, weights=self._weights_data)  # super in order not to save positions_rec_data
            self.wrap = wrap
            # Gaussian smoothing before density contrast calculation
            self.mesh_data.smooth_gaussian(self.smoothing_radius, method='fft')

        self.set_density_contrast(ran_min=self.ran_min, smoothing_radius=self.smoothing_radius)
        del self.mesh_data
        delta_k = self.mesh_delta.to_complex()
        del self.mesh_delta
        k = utils.broadcast_arrays(*delta_k.coords())
        delta_k.prod_sum([k**2 for k in delta_k.coords()], exp=-1)
        delta_k[0, 0, 0] = 0.
        # k = utils.broadcast_arrays(*delta_k.coords())
        # k2 = sum(kk**2 for kk in k)
        # k2[0,0,0] = 1. # to avoid dividing by 0
        # delta_k /= k2
        self.log_info('Computing displacement field.')
        shifts = np.empty_like(self._positions_rec_data)
        psis = []
        for iaxis in range(delta_k.ndim):
            # no need to compute psi on axis where los is 0
            if not return_psi and self.los is not None and self.los[iaxis] == 0:
                shifts[:, iaxis] = 0.
                continue
            psi = (delta_k * 1j * k[iaxis]).to_real()
            # Reading shifts at reconstructed data real-space positions
            shifts[:, iaxis] = psi.read_cic(self._positions_rec_data, wrap=True)
            if return_psi: psis.append(psi)

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
        diff = self._positions_rec_data - self.mesh_randoms.offset
        if (not self.wrap) and np.any((diff < 0) | (diff > self.mesh_randoms.boxsize - self.mesh_randoms.cellsize)):
            self.log_warning('Some particles are out-of-bounds.')
        self._iter += 1
        if return_psi:
            return psis

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

        def read_cic(positions, wrap=False):
            shifts = np.empty_like(positions)
            for iaxis, psi in enumerate(self.mesh_psi):
                shifts[:, iaxis] = psi.read_cic(positions, wrap=wrap)
            return shifts

        if isinstance(positions, str) and positions == 'data':
            # _positions_rec_data already wrapped during iteration
            shifts = read_cic(self._positions_rec_data, wrap=True)
            if field == 'disp':
                return shifts
            rsd = self._positions_data - self._positions_rec_data
            if field == 'rsd':
                return rsd
            # field == 'disp+rsd'
            shifts += rsd
            return shifts

        if self.wrap: positions = self.info.wrap(positions)  # wrap here for local los
        shifts = read_cic(positions, wrap=False)  # aleady wrapped

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
        # we follow convention of original algorithm: remove RSD first,
        # then remove Zeldovich displacement
        real_positions = positions - rsd
        diff = real_positions - self.mesh_psi[0].offset
        if (not self.wrap) and np.any((diff < 0) | (diff > self.mesh_psi[0].boxsize - self.mesh_psi[0].cellsize)):
            self.log_warning('Some particles are out-of-bounds.')
        shifts = read_cic(real_positions, wrap=True)

        return shifts + rsd

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
        shifts = self.read_shifts(positions, field=field)
        if isinstance(positions, str) and positions == 'data':
            positions = self._positions_data
        positions = positions - shifts
        if self.wrap: positions = self.info.wrap(positions)
        return positions


class IterativeFFTParticleReconstruction(OriginalIterativeFFTParticleReconstruction):

    """Any update / test / improvement upon original algorithm."""
