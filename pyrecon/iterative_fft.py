"""Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591) algorithm."""

from .recon import BaseReconstruction
from . import utils


class IterativeFFTReconstruction(BaseReconstruction):
    """
    Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591)
    field-level (as opposed to :class:`IterativeFFTParticleReconstruction`) algorithm.
    """
    _compressed = True
    _f_z = True
    _bias_z = True

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
            utils.safe_divide(slab, sum(kk**2 for kk in kslab), inplace=True)

        self.mesh_delta_real = self.mesh_delta.copy()
        # Now compute \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r}
        # In the plane-parallel case (self.los is a given vector), this is simply \beta IFFT((\hat{k} \cdot \hat{\eta})^{2} \delta(k))
        if self.los is not None:
            # global los
            disp_deriv_k = delta_k.copy()
            for kslab, slab in zip(disp_deriv_k.slabs.x, disp_deriv_k.slabs):
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
                    for kslab, islab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs.i, disp_deriv.slabs):
                        mask = (islab[iaxis] != self.nmesh[iaxis] // 2) & (islab[jaxis] != self.nmesh[jaxis] // 2)
                        mask |= (islab[iaxis] == self.nmesh[iaxis] // 2) & (islab[jaxis] == self.nmesh[jaxis] // 2)
                        slab[...] *= kslab[iaxis] * kslab[jaxis] * mask  # delta_k already divided by k^{2}
                    disp_deriv = disp_deriv.c2r()
                    for rslab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs):
                        rslab = self._transform_rslab(rslab)
                        slab[...] *= utils.safe_divide(rslab[iaxis] * rslab[jaxis], sum(rr**2 for rr in rslab))
                    factor = (1. + (iaxis != jaxis)) * self.beta  # we have j >= i and double-count j > i to account for j < i
                    if self._iter == 0:
                        # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                        factor /= (1. + self.beta)
                    # remove RSD part
                    self.mesh_delta_real -= factor * disp_deriv
        self._iter += 1

    def _compute_psi(self):
        # Compute Zeldovich displacements given reconstructed real space density
        delta_k = self.mesh_delta_real.r2c()
        psis = []
        for iaxis in range(delta_k.ndim):
            psi = delta_k.copy()
            for kslab, islab, slab in zip(psi.slabs.x, psi.slabs.i, psi.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2
                slab[...] *= 1j * utils.safe_divide(kslab[iaxis], sum(kk**2 for kk in kslab)) * mask
            psis.append(psi.c2r())
            del psi
        return psis

class HybridIFFTReconstruction(IterativeFFTReconstruction):
    """
    Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591)
    field-level (as opposed to :class:`IterativeFFTParticleReconstruction`) algorithm.
    """
    _compressed = True
    _f_z = True
    _bias_z = True

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
        self._positions_rec_data = self._positions_data.copy()
        
        for iter in range(niterations):
            self._iterate(return_psi=(iter == niterations - 1))
        del self.mesh_delta
        self.mesh_psi = self._compute_psi()
        del self.mesh_delta_real

    def _iterate(self, return_psi=False):
        if self.mpicomm.rank == 0:
            self.log_info('Running iteration {:d}.'.format(self._iter))
        # This is an implementation of eq. 22 and 24 in https://arxiv.org/pdf/1504.02591.pdf
        # \delta_{g,\mathrm{real},n} is self.mesh_delta_real
        # \delta_{g,\mathrm{red}} is self.mesh_delta
        # First compute \delta(k)/k^{2} based on current \delta_{g,\mathrm{real},n} to estimate \phi_{\mathrm{est},n} (eq. 24)
        delta_k = self.mesh_delta_real.r2c()
        for kslab, slab in zip(delta_k.slabs.x, delta_k.slabs):
            utils.safe_divide(slab, sum(kk**2 for kk in kslab), inplace=True)

        self.mesh_delta_real = self.mesh_delta.copy()
        # Now compute \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r}
        # In the plane-parallel case (self.los is a given vector), this is simply \beta IFFT((\hat{k} \cdot \hat{\eta})^{2} \delta(k))
        if self.los is not None:
            # global los
            disp_deriv_k = delta_k.copy()
            for kslab, slab in zip(disp_deriv_k.slabs.x, disp_deriv_k.slabs):
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
                    for kslab, islab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs.i, disp_deriv.slabs):
                        mask = (islab[iaxis] != self.nmesh[iaxis] // 2) & (islab[jaxis] != self.nmesh[jaxis] // 2)
                        mask |= (islab[iaxis] == self.nmesh[iaxis] // 2) & (islab[jaxis] == self.nmesh[jaxis] // 2)
                        slab[...] *= kslab[iaxis] * kslab[jaxis] * mask  # delta_k already divided by k^{2}
                    disp_deriv = disp_deriv.c2r()
                    for rslab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs):
                        rslab = self._transform_rslab(rslab)
                        slab[...] *= utils.safe_divide(rslab[iaxis] * rslab[jaxis], sum(rr**2 for rr in rslab))
                    factor = (1. + (iaxis != jaxis)) * self.beta  # we have j >= i and double-count j > i to account for j < i
                    if self._iter == 0:
                        # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                        factor /= (1. + self.beta)
                    # remove RSD part
                    self.mesh_delta_real -= factor * disp_deriv

        # =================================================
        # Initialize an array to store displacement shifts for each particle in the reconstructed data space.
        shifts = np.empty_like(self._positions_rec_data)

        # This list is used to optionally store the intermediate displacement fields 
        psis = []

        # Loop over each spatial dimension (x, y, z) to compute shifts along each axis.
        for iaxis in range(delta_k.ndim):

            # No need to compute psi on axis where los is 0
            if not return_psi and self.los is not None and self.los[iaxis] == 0:
                shifts[:, iaxis] = 0.
                continue

            # Create a copy of the Fourier-space density field `delta_k` to modify and extract displacement components.
            psi = delta_k.copy()

            # Apply Fourier-space operations to extract displacements along the current axis.
            for kslab, islab, slab in zip(psi.slabs.x, psi.slabs.i, psi.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2  
                slab[...] *= 1j * kslab[iaxis] * mask  

            psi = psi.c2r()

            # Read out displacement shifts at the reconstructed particle positions (`_positions_rec_data`).
            # This interpolates the computed displacement field to find the displacements at the actual particle locations.
            shifts[:, iaxis] = self._readout(psi, self._positions_rec_data)

            # If requested, store the displacement field (psi) for further analysis.
            if return_psi:
                psis.append(psi)

            del psi

        # If `self.los` is not explicitly set, compute a normalized los vector from the particle positions.
        if self.los is None:
            los = utils.safe_divide(self._positions_data, utils.distance(self._positions_data)[:, None])
        else:
            los = self.los

        # Adjust shifts for the first iteration using a correction to remove Redshift-Space Distortion (RSD) effects.
        # This follows Eq. 12 from Burden et al. (2015), which applies a correction factor to improve convergence.
        if self._iter == 0:
            shifts -= self.beta / (1 + self.beta) * np.sum(shifts * los, axis=-1)[:, None] * los

        # **New Reconstruction Step**
        # Rather than keeping particle positions static throughout iterations, we iteratively update the reconstructed positions.
        # This step modifies `self._positions_rec_data` by subtracting an RSD-related displacement correction.
        # The correction term scales the computed shifts along the los by a factor `self.f`, refining reconstructed positions.
        self._positions_rec_data = self._positions_data - self.f * np.sum(shifts * los, axis=-1)[:, None] * los

        self._iter += 1

    def _compute_psi(self):
        # Compute Zeldovich displacements given reconstructed real space density
        delta_k = self.mesh_delta_real.r2c()
        psis = []
        for iaxis in range(delta_k.ndim):
            psi = delta_k.copy()
            for kslab, islab, slab in zip(psi.slabs.x, psi.slabs.i, psi.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2
                slab[...] *= 1j * utils.safe_divide(kslab[iaxis], sum(kk**2 for kk in kslab)) * mask
            psis.append(psi.c2r())
            del psi
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
            los = utils.safe_divide(positions, utils.distance(positions)[:, None])
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

