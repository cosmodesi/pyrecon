"""Implementation of base reconstruction class."""

import inspect
import functools

import numpy as np

from pmesh.pm import ParticleMesh

from .mesh import _get_mesh_attrs, _get_resampler, _wrap_positions
from .utils import BaseClass
from . import utils, mpi


def _gaussian_kernel(smoothing_radius):

    def kernel(k, v):
        return v * np.exp(- 0.5 * sum((ki * smoothing_radius)**2 for ki in k))

    return kernel


def _get_real_dtype(dtype):
    # Return real-dtype equivalent
    return np.empty(0, dtype=dtype).real.dtype


def _format_positions(positions, position_type='xyz', dtype=None, copy=True, mpicomm=None, mpiroot=None):
    # Format input array of positions
    # position_type in ["xyz", "rdd", "pos"]

    def __format_positions(positions):
        if position_type == 'pos':  # array of shape (N, 3)
            positions = np.array(positions, dtype=dtype, copy=copy)
            if not np.issubdtype(positions.dtype, np.floating):
                return None, 'Input position arrays should be of floating type, not {}'.format(positions.dtype)
            if positions.shape[-1] != 3:
                return None, 'For position type = {}, please provide a (N, 3) array for positions'.format(position_type)
            return positions, None
        # Array of shape (3, N)
        positions = list(positions)
        for ip, p in enumerate(positions):
            # Cast to the input dtype if exists (may be set by previous weights)
            positions[ip] = np.asarray(p, dtype=dtype)
        size = len(positions[0])
        dt = positions[0].dtype
        if not np.issubdtype(dt, np.floating):
            return None, 'Input position arrays should be of floating type, not {}'.format(dt)
        for p in positions[1:]:
            if len(p) != size:
                return None, 'All position arrays should be of the same size'
            if p.dtype != dt:
                return None, 'All position arrays should be of the same type, you can e.g. provide dtype'
        if len(positions) != 3:
            return None, 'For position type = {}, please provide a list of 3 arrays for positions (found {:d})'.format(position_type, len(positions))
        if position_type == 'rdd':  # RA, Dec, distance
            positions = utils.sky_to_cartesian(positions[2], *positions[:2], degree=True).T
        elif position_type != 'xyz':
            return None, 'Position type should be one of ["pos", "xyz", "rdd"]'
        return np.asarray(positions).T, None

    error = None
    if mpiroot is None or (mpicomm.rank == mpiroot):
        if positions is not None and (position_type == 'pos' or not all(position is None for position in positions)):
            positions, error = __format_positions(positions)  # return error separately to raise on all processes
    if mpicomm is not None:
        error = mpicomm.allgather(error)
    else:
        error = [error]
    errors = [err for err in error if err is not None]
    if errors:
        raise ValueError(errors[0])
    if mpiroot is not None and mpicomm.bcast(positions is not None if mpicomm.rank == mpiroot else None, root=mpiroot):
        positions = mpi.scatter(positions, mpicomm=mpicomm, mpiroot=mpiroot)
    return positions


def _format_weights(weights, size=None, dtype=None, copy=True, mpicomm=None, mpiroot=None):
    # Format input weights.
    def __format_weights(weights):
        if weights is None:
            return weights
        weights = weights.astype(dtype, copy=copy)
        return weights

    weights = __format_weights(weights)
    if mpiroot is None:
        is_none = mpicomm.allgather(weights is None)
        if any(is_none) and not all(is_none):
            raise ValueError('mpiroot = None but weights are None on some ranks')
    else:
        weights = mpi.scatter(weights, mpicomm=mpicomm, mpiroot=mpiroot)

    if size is not None and weights is not None and len(weights) != size:
        raise ValueError('Weight arrays should be of the same size as position arrays')
    return weights


def format_positions_weights_wrapper(func):
    """Method wrapper applying _format_positions and _format_weigths on input."""
    @functools.wraps(func)
    def wrapper(self, positions, weights=None, copy=False, **kwargs):
        position_type = kwargs.pop('position_type', self.position_type)
        mpiroot = kwargs.pop('mpiroot', self.mpiroot)
        positions = _format_positions(positions, position_type=position_type, copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)
        if not self.wrap:
            low, high = self.boxcenter - self.boxsize / 2., self.boxcenter + self.boxsize / 2.
            if any(self.mpicomm.allgather(np.any((positions < low) | (positions > high)))):
                raise ValueError('positions not in box range {} - {}'.format(low, high))
        weights = _format_weights(weights, size=len(positions), copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)
        return func(self, positions=positions, weights=weights, **kwargs)
    return wrapper


def format_positions_wrapper(return_input_type=True):
    def format_positions(func):
        """Method wrapper applying _format_positions on input, and gathering result on mpiroot."""
        @functools.wraps(func)
        def wrapper(self, positions, copy=False, **kwargs):
            position_type = kwargs.pop('position_type', self.position_type)
            mpiroot = kwargs.pop('mpiroot', self.mpiroot)
            if not all(self.mpicomm.allgather(isinstance(positions, str))):  # for IterativeFFTParticleReconstruction
                positions = _format_positions(positions, position_type=position_type, copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)
                if not self.wrap:
                    low, high = self.boxcenter - self.boxsize / 2., self.boxcenter + self.boxsize / 2.
                    if any(self.mpicomm.allgather(np.any((positions < low) | (positions > high)))):
                        raise ValueError('positions not in box range {} - {}'.format(low, high))
            toret = func(self, positions=positions, **kwargs)
            if toret is not None and mpiroot is not None:  # positions returned, gather on the same rank
                toret = mpi.gather(toret, mpicomm=self.mpicomm, mpiroot=mpiroot)
            if toret is not None and return_input_type:
                if position_type == 'rdd':
                    dist, ra, dec = utils.cartesian_to_sky(toret)
                    toret = [ra, dec, dist]
                elif position_type == 'xyz':
                    toret = toret.T
            return toret
        return wrapper
    return format_positions


class ReconstructionError(Exception):

    """Error raised when issue with reconstruction."""


class BaseReconstruction(BaseClass):
    """
    Base template reconstruction class.
    Reconstruction algorithms should extend this class, by (at least) implementing:

    - :meth:`run`

    A standard reconstruction would be:

    .. code-block:: python

        # MyReconstruction is your reconstruction algorithm
        recon = MyReconstruction(f=0.8, bias=2.0, nmesh=512, boxsize=1000., boxcenter=2000.)
        recon.assign_data(positions_data, weights_data)
        recon.assign_randoms(positions_randoms, weights_randoms)
        recon.set_density_contrast()
        recon.run()
        positions_rec_data = recon.read_shifted_positions(positions_data)
        # RecSym = remove large scale RSD from randoms
        positions_rec_randoms = recon.read_shifted_positions(positions_randoms)
        # Or RecIso
        # positions_rec_randoms = recon.read_shifted_positions(positions_randoms, field='disp')

    Attributes
    ----------
    mesh_data : RealField
        Mesh (3D grid) to assign ("paint") galaxies to.

    mesh_randoms : RealField
        Mesh (3D grid) to assign ("paint") randoms to.

    f : float
        Growth rate.

    bias : float
        Galaxy bias.

    los : array
        If ``None``, local (varying) line-of-sight.
        Else line-of-sight (unit) 3-vector.

    boxsize, boxcenter, cellsize, offset : array
        Mesh properties.
    """
    _slab_npoints_max = int(1024 * 1024 * 4)
    _compressed = False

    def __init__(self, f=None, bias=None, los=None, nmesh=None, boxsize=None, boxcenter=None, cellsize=None, boxpad=2., wrap=False,
                 data_positions=None, randoms_positions=None, data_weights=None, randoms_weights=None,
                 positions=None, position_type='pos', resampler='cic', decomposition=None, fft_plan='estimate', dtype='f8', mpiroot=None, mpicomm=mpi.COMM_WORLD, **kwargs):
        """
        Initialize :class:`BaseReconstruction`.

        Parameters
        ----------
        f : float, default=None.
            Growth rate.

        bias : float, default=None.
            Galaxy bias.

        los : string, array_like, default=None
            If ``los`` is ``None``, use local (varying) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        nmesh : array, int, default=None
            Mesh size, i.e. number of mesh nodes along each axis.

        boxsize : array, float, default=None
            Physical size of the box along each axis, defaults to maximum extent taken by all input positions, times ``boxpad``.

        boxcenter : array, float, default=None
            Box center, defaults to center of the Cartesian box enclosing all input positions.

        cellsize : array, float, default=None
            Physical size of mesh cells.
            If not ``None``, and mesh size ``nmesh`` is not ``None``, used to set ``boxsize`` as ``nmesh * cellsize``.
            If ``nmesh`` is ``None``, it is set as (the nearest integer(s) to) ``boxsize / cellsize``.

        boxpad : float, default=2.
            When ``boxsize`` is determined from input positions, take ``boxpad`` times the smallest box enclosing positions as ``boxsize``.

        wrap : bool, default=False
            Whether to wrap input positions in [0, boxsize]?
            If ``False`` and input positions do not fit in the the box size, raise a :class:`ValueError`.

        positions : list, array, default=None
            Optionally, positions used to defined box size. Of shape (3, N) or (N, 3), depending on ``position_type``.

        data_positions : list, array, default=None
            Positions in the data catalog. Of shape (3, N) or (N, 3), depending on ``position_type``.
            If provided, reconstruction will be run directly (and ``data_positions`` will be added to ``positions`` to define the box size).

        randoms_positions : list, array, default=None
            Positions in the randoms catalog. See ``data_positions``.

        data_weights : array of shape (N,), default=None
            Optionally, weights in the data catalog.

        randoms_weights : array of shape (N,), default=None
            Optionally, weights in the randoms catalog.

        position_type : string, default='xyz'
            Type of input positions, one of:

                - "pos": Cartesian positions of shape (N, 3)
                - "xyz": Cartesian positions of shape (3, N)
                - "rdd": RA/Dec in degree, distance of shape (3, N)

        fft_plan : string, default='estimate'
            FFT planning. 'measure' may allow for faster FFTs, but is slower to set up than 'estimate'.

        resampler : string, ResampleWindow, default='tsc'
            Resampler used to assign particles to the mesh.
            Choices are ['ngp', 'cic', 'tcs', 'pcs'].

        dtype : string, dtype, default='f8'
            The data type to use for the mesh.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=MPI.COMM_WORLD
            The MPI communicator.
        """
        self.mpicomm = mpicomm
        self.mpiroot = mpiroot
        self.position_type = position_type
        self.wrap = bool(wrap)
        self.rdtype = _get_real_dtype(dtype)
        self.dtype = np.dtype(dtype) if self._compressed else 'c{:d}'.format(2 * self.rdtype.itemsize)
        self.f = f
        self.bias = bias
        positions = _format_positions(positions, position_type=self.position_type, copy=True, mpicomm=self.mpicomm, mpiroot=self.mpiroot)
        data_positions = _format_positions(data_positions, position_type=self.position_type, copy=True, mpicomm=self.mpicomm, mpiroot=self.mpiroot)
        randoms_positions = _format_positions(randoms_positions, position_type=self.position_type, copy=True, mpicomm=self.mpicomm, mpiroot=self.mpiroot)
        all_positions = [pos for pos in [positions, data_positions, randoms_positions] if pos is not None]
        self.nmesh, self.boxsize, self.boxcenter = _get_mesh_attrs(nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, cellsize=cellsize, positions=all_positions,
                                                                   boxpad=boxpad, check=positions is not None and not self.wrap, mpicomm=self.mpicomm)
        self.resampler = _get_resampler(resampler=resampler)
        self.pm = ParticleMesh(BoxSize=self.boxsize, Nmesh=self.nmesh, dtype=self.dtype, comm=self.mpicomm, resampler=self.resampler, np=decomposition, plan_method=fft_plan)
        self.set_los(los)
        if self.mpicomm.rank == 0:
            self.log_info('Using mesh with nmesh={}, boxsize={}, boxcenter={}.'.format(self.nmesh, self.boxsize, self.boxcenter))
        if data_positions is not None:
            data_weights = _format_weights(data_weights, size=len(data_positions), copy=True, mpicomm=self.mpicomm, mpiroot=self.mpiroot)
            self.assign_data(data_positions, data_weights, position_type='pos', copy=False, mpiroot=None)
            if randoms_positions is not None:
                randoms_weights = _format_weights(randoms_weights, size=len(randoms_positions), copy=True, mpicomm=self.mpicomm, mpiroot=self.mpiroot)
                self.assign_randoms(randoms_positions, randoms_weights, position_type='pos', copy=False, mpiroot=None)
            run_kwargs = {name: kwargs.pop(name) for name in list(kwargs.keys()) if name not in inspect.getargspec(self.set_density_contrast).args}
            self.set_density_contrast(**kwargs)
            self.run(**run_kwargs)

    @property
    def beta(self):
        r""":math:`\beta` parameter, as the ratio of the growth rate to the galaxy bias."""
        return self.f / self.bias

    def set_cosmo(self, f=None, bias=None, beta=None):
        r"""
        Set cosmology.

        Parameters
        ----------
        f : float
            Growth rate. If ``None`` and `beta`` is provided, set ``f`` as ``beta * bias``;
            else growth rate is left unchanged.

        bias : float
            Bias. If ``None``, bias is left unchanged.

        beta : float
            :math:`\beta` parameter. If not ``None``, overrides ``f`` as ``beta * bias``.
        """
        if bias is None:
            bias = self.bias
        if beta is not None: f = beta * bias
        if f is None:
            f = self.f
        self.f = f
        self.bias = bias
        if self.f is None:
            raise ValueError('Provide f')
        if self.bias is None:
            raise ValueError('Provide bias')

    def set_los(self, los=None):
        """
        Set line-of-sight.

        Parameters
        ----------
        los : string, array
            If ``los`` is ``None``, use local (varying) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.
        """
        if los in [None, 'local']:
            self.los = None
        else:
            if isinstance(los, str):
                los = 'xyz'.index(los)
            if np.ndim(los) == 0:
                ilos = los
                los = np.zeros(3, dtype='f8')
                los[ilos] = 1.
            los = np.array(los, dtype='f8')
            self.los = los / utils.distance(los)

    @property
    def cellsize(self):
        return self.boxsize / self.nmesh

    @property
    def offset(self):
        return self.boxcenter - self.boxsize / 2.

    def _wrap(self, positions):
        return _wrap_positions(positions, self.boxsize, self.offset)

    def _transform_rslab(self, rslab):
        toret = []
        for ii, rr in enumerate(rslab):
            mask = rr < 0.
            rr = rr.copy()
            rr[mask] += self.boxsize[ii]
            rr += self.offset[ii]
            toret.append(rr)
        return toret

    def _paint(self, positions, weights=None, out=None, transform=None):
        positions = positions - self.offset
        scalar_weights = weights is None
        if not all(self.mpicomm.allgather(np.isfinite(positions).all())):
            raise ValueError('Some positions are NaN/inf')
        if not scalar_weights and not all(self.mpicomm.allgather(np.isfinite(weights).all())):
            raise ValueError('Some weights are NaN/inf')
        if out is None:
            out = self.pm.create('real', value=0.)

        # We work by slab to limit memory footprint
        # Merely copy-pasted from https://github.com/bccp/nbodykit/blob/4aec168f176939be43f5f751c90363b39ec6cf3a/nbodykit/source/mesh/catalog.py#L300
        def paint_slab(sl):
            # Decompose positions such that they live in the same region as the mesh in the current process
            p = positions[sl]
            size = len(p)
            layout = self.pm.decompose(p, smoothing=self.resampler.support)
            # If we are receiving too many particles, abort and retry with a smaller chunksize
            recvlengths = self.pm.comm.allgather(layout.recvlength)
            if any(recvlength > 2 * self._slab_npoints_max for recvlength in recvlengths):
                if self.pm.comm.rank == 0:
                    self.log_info('Throttling slab size as some ranks will receive too many particles. ({:d} > {:d})'.format(max(recvlengths), self._slab_npoints_max * 2))
                raise StopIteration
            p = layout.exchange(p)
            w = weights if scalar_weights else layout.exchange(weights[sl])
            # hold = True means no zeroing of out
            self.pm.paint(p, mass=w, resampler=self.resampler, transform=transform, hold=True, out=out)
            return size

        islab = 0
        slab_npoints = self._slab_npoints_max
        sizes = self.pm.comm.allgather(len(positions))
        csize = sum(sizes)
        local_size_max = max(sizes)
        painted_size = 0

        import gc
        while islab < local_size_max:

            sl = slice(islab, islab + slab_npoints)

            if self.pm.comm.rank == 0:
                self.log_info('Slab {:d} ~ {:d} / {:d}.'.format(islab, islab + slab_npoints, local_size_max))
            try:
                painted_size_slab = paint_slab(sl)
            except StopIteration:
                slab_npoints = slab_npoints // 2
                if slab_npoints < 1:
                    raise RuntimeError('Cannot find a slab size that fits into memory.')
                continue
            finally:
                # collect unfreed items
                gc.collect()

            painted_size += self.pm.comm.allreduce(painted_size_slab)

            if self.pm.comm.rank == 0:
                self.log_info('Painted {:d} out of {:d} objects to mesh.'.format(painted_size, csize))

            islab += slab_npoints
            slab_npoints = min(self._slab_npoints_max, int(slab_npoints * 1.2))
        return out

    def _readout(self, mesh, positions):
        positions = positions - self.offset
        layout = self.pm.decompose(positions, smoothing=self.resampler)
        positions = layout.exchange(positions)
        values = mesh.readout(positions, resampler=self.resampler)
        return layout.gather(values, mode='sum', out=None)

    def _smooth_gaussian(self, mesh):
        mesh = mesh.r2c()
        mesh.apply(_gaussian_kernel(self.smoothing_radius), kind='wavenumber', out=Ellipsis)
        return mesh.c2r()

    @format_positions_weights_wrapper
    def assign_data(self, positions, weights=None, **kwargs):
        """
        Assign (paint) data to :attr:`mesh_data`.
        This can be done slab-by-slab (e.g. to reduce memory footprint).

        Parameters
        ----------
        positions : array of shape (N, 3)
            Cartesian positions.

        weights : array of shape (N,), default=None
            Weights; default to 1.
        """
        if not hasattr(self, 'mesh_data'):
            self.mesh_data = self.pm.create(type='real', value=0.)
        self._paint(positions, weights=weights, out=self.mesh_data)

    @format_positions_weights_wrapper
    def assign_randoms(self, positions, weights=None):
        """Same as :meth:`assign_data`, but for random objects."""
        if not hasattr(self, 'mesh_randoms'):
            self.mesh_randoms = self.pm.create(type='real', value=0.)
            self._size_randoms = 0
        self._paint(positions, weights=weights, out=self.mesh_randoms)
        self._size_randoms += self.mpicomm.allreduce(len(positions))

    @property
    def has_randoms(self):
        return hasattr(self, 'mesh_randoms')

    def set_density_contrast(self, ran_min=0.01, smoothing_radius=15., check=False):
        r"""
        Set :math:`\delta` field :attr:`mesh_delta` from data and randoms fields :attr:`mesh_data` and :attr:`mesh_randoms`.

        Note
        ----
        This method follows Julian's reconstruction code.

        Parameters
        ----------
        ran_min : float, default=0.01
            :attr:`mesh_randoms` points below this threshold times mean random weights have their density contrast set to 0.

        smoothing_radius : float, default=15
            Smoothing scale.

        check : bool, default=False
            If ``True``, run some tests (printed in logger) to assess whether enough randoms have been used.
        """
        self.smoothing_radius = smoothing_radius
        self.mesh_delta = self._smooth_gaussian(self.mesh_data)
        del self.mesh_data

        if self.has_randoms:
            if check:
                nnonzero = self.mpicomm.allreduce(sum(np.sum(randoms > 0.) for randoms in self.mesh_randoms))
                if nnonzero < 2: raise ValueError('Very few randoms!')

            self.mesh_randoms = self._smooth_gaussian(self.mesh_randoms)

            sum_data, sum_randoms = self.mesh_delta.csum(), self.mesh_randoms.csum()
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

    def run(self, *args, **kwargs):
        """Run reconstruction; to be implemented in your algorithm."""
        raise NotImplementedError('Implement method "run" in your "{}"-inherited algorithm'.format(self.__class__.___name__))

    @format_positions_wrapper(return_input_type=False)
    def read_shifts(self, positions, field='disp+rsd'):
        """
        Read displacement at input positions.
        To get shifted/reconstructed positions, given reconstruction instance ``recon``:

        .. code-block:: python

            positions_rec_data = positions_data - recon.read_shifts(positions_data)
            # RecSym = remove large scale RSD from randoms
            positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms)
            # Or RecIso
            # positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms, field='disp')

        Or directly use :meth:`read_shifted_positions` (which wraps output positions if :attr:`wrap`).

        Parameters
        ----------
        positions : array of shape (N, 3)
            Cartesian positions.

        field : string, default='disp+rsd'
            Either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        kwargs : dict
            ``position_type``, ``mpiroot`` can be provided to override default :attr:`position_type`, :attr:`mpiroot`.

        Returns
        -------
        shifts : array of shape (N, 3)
            Displacements.
        """
        field = field.lower()
        allowed_fields = ['disp', 'rsd', 'disp+rsd']
        if field not in allowed_fields:
            raise ReconstructionError('Unknown field {}. Choices are {}'.format(field, allowed_fields))

        shifts = np.empty_like(positions)
        for iaxis, psi in enumerate(self.mesh_psi):
            shifts[:, iaxis] = self._readout(psi, positions)
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

    @format_positions_wrapper(return_input_type=True)
    def read_shifted_positions(self, positions, field='disp+rsd'):
        """
        Read shifted positions i.e. the difference ``positions - self.read_shifts(positions, field=field)``.
        Output (and input) positions are wrapped if :attr:`wrap`.

        Parameters
        ----------
        positions : list, array
            Positions of shape (3, N) or (N, 3), depending on ``position_type``.

        field : string, default='disp+rsd'
            Apply either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        kwargs : dict
            ``position_type``, ``mpiroot`` can be provided to override default :attr:`position_type`, :attr:`mpiroot`.

        Returns
        -------
        positions : list, array
            Shifted positions, of same type as input.
        """
        shifts = self.read_shifts(positions, field=field, position_type='pos', mpiroot=None)
        positions = positions - shifts
        if self.wrap: positions = self._wrap(positions)
        return positions
