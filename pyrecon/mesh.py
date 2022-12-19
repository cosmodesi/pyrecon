import numpy as np
from pmesh.window import FindResampler, ResampleWindow

from .utils import _get_box, _make_array
from . import mpi


def _get_resampler(resampler):
    # Return :class:`ResampleWindow` from string or :class:`ResampleWindow` instance
    if isinstance(resampler, ResampleWindow):
        return resampler
    conversions = {'ngp': 'nnb', 'cic': 'cic', 'tsc': 'tsc', 'pcs': 'pcs'}
    if resampler not in conversions:
        raise ValueError('Unknown resampler {}, choices are {}'.format(resampler, list(conversions.keys())))
    resampler = conversions[resampler]
    return FindResampler(resampler)


def _get_resampler_name(resampler):
    # Translate input :class:`ResampleWindow` instance to string
    conversions = {'nearest': 'ngp', 'tunednnb': 'ngp', 'tunedcic': 'cic', 'tunedtsc': 'tsc', 'tunedpcs': 'pcs'}
    return conversions[resampler.kind]


def _wrap_positions(positions, boxsize, offset=0.):
    return np.asarray((positions - offset) % boxsize + offset, dtype=positions.dtype)


def _get_mesh_attrs(nmesh=None, boxsize=None, boxcenter=None, cellsize=None, positions=None, boxpad=1.5, check=True, select_nmesh=None, mpicomm=mpi.COMM_WORLD):
    """
    Compute enclosing box.

    Parameters
    ----------
    nmesh : array, int, default=None
        Mesh size, i.e. number of mesh nodes along each axis.
        If not provided, see ``value``.

    boxsize : float, default=None
        Physical size of the box.
        If not provided, see ``positions``.

    boxcenter : array, float, default=None
        Box center.
        If not provided, see ``positions``.

    cellsize : array, float, default=None
        Physical size of mesh cells.
        If not ``None``, ``boxsize`` is ``None`` and mesh size ``nmesh`` is not ``None``, used to set ``boxsize`` to ``nmesh * cellsize``.
        If ``nmesh`` is ``None``, it is set to (the nearest integer(s) to) ``boxsize / cellsize`` if ``boxsize`` is provided,
        else to the nearest even integer to ``boxsize / cellsize``, and ``boxsize`` is then reset to ``nmesh * cellsize``.

    positions : (list of) (N, 3) arrays, default=None
        If ``boxsize`` and / or ``boxcenter`` is ``None``, use this (list of) position arrays
        to determine ``boxsize`` and / or ``boxcenter``.

    boxpad : float, default=1.5
        When ``boxsize`` is determined from ``positions``, take ``boxpad`` times the smallest box enclosing ``positions`` as ``boxsize``.

    check : bool, default=True
        If ``True``, and input ``positions`` (if provided) are not contained in the box, raise a :class:`ValueError`.

    select_nmesh : callable, default=True
        Function that takes in a 3-array ``nmesh``, and returns the 3-array ``nmesh``.
        Used by :class:`MultiGridReconstruction` to select mesh sizes compatible with the algorithm.

    mpicomm : MPI communicator, default=MPI.COMM_WORLD
        The MPI communicator.

    Returns
    -------
    nmesh : array of shape (3,)
        Mesh size, i.e. number of mesh nodes along each axis.

    boxsize : array
        Physical size of the box.

    boxcenter : array
        Box center.
    """
    if boxsize is None or boxcenter is None or check:
        if positions is None:
            raise ValueError('positions must be provided if boxsize and boxcenter are not specified, or check is True')
        if not isinstance(positions, (tuple, list)):
            positions = [positions]
        positions = [pos for pos in positions if pos is not None]
        # Find bounding coordinates
        if mpicomm.allreduce(sum(pos.shape[0] for pos in positions)) <= 1:
            raise ValueError('<= 1 particles found; cannot infer boxsize or boxcenter')
        pos_min, pos_max = _get_box(*positions)
        pos_min, pos_max = np.min(mpicomm.allgather(pos_min), axis=0), np.max(mpicomm.allgather(pos_max), axis=0)
        delta = np.abs(pos_max - pos_min)
        if boxcenter is None: boxcenter = 0.5 * (pos_min + pos_max)
        if boxsize is None:
            if cellsize is not None and nmesh is not None:
                boxsize = nmesh * cellsize
            else:
                boxsize = delta.max() * boxpad
        if check and (boxsize < delta).any():
            raise ValueError('boxsize {} too small to contain all data (max {})'.format(boxsize, delta))

    boxsize = _make_array(boxsize, 3, dtype='f8')
    if nmesh is None:
        if cellsize is not None:
            cellsize = _make_array(cellsize, 3, dtype='f8')
            nmesh = boxsize / cellsize
            nmesh = np.ceil(nmesh).astype('i8')
            nmesh += nmesh % 2  # to make it even
            if select_nmesh is not None: nmesh = select_nmesh(nmesh)
            boxsize = nmesh * cellsize  # enforce exact cellsize
        else:
            raise ValueError('nmesh (or cellsize) must be specified')
    nmesh = _make_array(nmesh, 3, dtype='i4')
    if select_nmesh is not None:
        recommended_nmesh = select_nmesh(nmesh)
        if not np.all(recommended_nmesh == nmesh):
            import warnings
            warnings.warn('Recommended nmesh is {}, provided nmesh is {}'.format(recommended_nmesh, nmesh))
    boxcenter = _make_array(boxcenter, 3, dtype='f8')
    if np.any(nmesh % 2):
        raise NotImplementedError('Odd sizes not supported by pmesh for now')
    return nmesh, boxsize, boxcenter
