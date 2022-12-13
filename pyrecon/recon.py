"""Implementation of base reconstruction class."""

import numpy as np

from .mesh import RealMesh
from .utils import BaseClass
from . import utils


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
    mesh_data : RealMesh
        Mesh (3D grid) to assign ("paint") galaxies to.

    mesh_randoms : RealMesh
        Mesh (3D grid) to assign ("paint") randoms to.

    f : float
        Growth rate.

    bias : float
        Galaxy bias.

    los : array
        If ``None``, local (varying) line-of-sight.
        Else line-of-sight (unit) 3-vector.

    info : MeshInfo
        Mesh information (boxsize, boxcenter, nmesh, etc.).

    boxsize, boxcenter, cellsize, offset : array
        Mesh properties; see :class:`MeshInfo`.
    """
    def __init__(self, f=0., bias=1., los=None, fft_engine='numpy', fft_wisdom=None, save_fft_wisdom=None, fft_plan='measure', wrap=False, **kwargs):
        """
        Initialize :class:`BaseReconstruction`.

        Parameters
        ----------
        f : float
            Growth rate.

        bias : float
            Galaxy bias.

        los : string, array_like, default=None
            If ``los`` is ``None``, use local (varying) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        fft_engine : string, BaseFFTEngine, default='numpy'
            Engine for fast Fourier transforms. See :class:`BaseFFTEngine`.
            We strongly recommend using 'fftw' for multithreaded FFTs.

        fft_wisdom : string, tuple, default=None
            Optionally, wisdom for FFTW, if ``fft_engine`` is 'fftw'.
            If a string, should be a path to previously saved FFT wisdom (with :func:`numpy.save`).
            If a tuple, directly corresponds to the wisdom.
            By default the wisdom given in ``save_fft_wisdom`` will be loaded, if exists.

        save_fft_wisdom : bool, string, default=None
            If not ``None``, path where to save the wisdom for FFTW.
            If ``True``, the wisdom will be saved in the default path: f'wisdom.shape-{nmesh[0]}-{nmesh[1]}-{nmesh[2]}.type-{type}.nthreads-{nthreads}.npy'.

        fft_plan : string, default='measure'
            Only used for FFTW. Choices are ['estimate', 'measure', 'patient', 'exhaustive'].
            The increasing amount of effort spent during the planning stage to create the fastest possible transform.
            Usually 'measure' is a good compromise.

        wrap : boolean, default=False
            If ``True``, wrap input particle positions into the box.

        kwargs : dict
            Arguments to build :attr:`mesh_data`, :attr:`mesh_randoms` (see :class:`RealMesh`).
        """
        self.set_cosmo(f=f, bias=bias)
        self.wrap = wrap
        self.mesh_data = RealMesh(**kwargs)
        self.mesh_randoms = RealMesh(**kwargs)
        # record mesh boxsize, cellsize and offset for later use when the meshes themselves get deleted
        self.info = self.mesh_randoms.info
        self.set_los(los)
        self.log_info('Using mesh {}.'.format(self.mesh_data))
        self.log_info('Using {:d} threads.'.format(self.mesh_data.nthreads))
        self.mesh_data.set_fft_engine(fft_engine, wisdom=fft_wisdom, save_wisdom=save_fft_wisdom, plan=fft_plan, hermitian=False)
        self.mesh_randoms.set_fft_engine(self.mesh_data.fft_engine)
        # Allow the wisdom to be accessed from outside if necessary
        self.fft_wisdom = getattr(self.mesh_data.fft_engine, 'wisdom', None)

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
        if bias is None: bias = self.bias
        if beta is not None: f = beta * bias
        if f is None: f = self.f
        self.f = f
        self.bias = bias

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
                los = np.zeros(3, dtype=self.mesh_data.dtype)
                los[ilos] = 1.
            los = np.array(los, dtype=self.mesh_data.dtype)
            self.los = los / utils.distance(los)

    def assign_data(self, positions, weights=None):
        """
        Assign (paint) data to :attr:`mesh_data`.
        This can be done slab-by-slab (e.g. to reduce memory footprint).

        Parameters
        ----------
        positions : array of shape (N,3)
            Cartesian positions.

        weights : array of shape (N,), default=None
            Weights; default to 1.
        """
        self.mesh_data.assign_cic(positions, weights=weights, wrap=self.wrap)

    def assign_randoms(self, positions, weights=None):
        """Same as :meth:`assign_data`, but for random objects."""
        if self.mesh_randoms.value is None:
            self._size_randoms = 0
        self.mesh_randoms.assign_cic(positions, weights=weights, wrap=self.wrap)
        self._size_randoms += len(positions)

    @property
    def has_randoms(self):
        return self.mesh_randoms.value is not None

    def set_density_contrast(self, ran_min=0.01, smoothing_radius=15., check=False, **kwargs):
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
            Smoothing scale, see :meth:`RealMesh.smooth_gaussian`.

        check : bool, default=False
            If ``True``, run some tests (printed in logger) to assess whether enough randoms have been used.

        kwargs : dict
            Optional arguments for :meth:`RealMesh.smooth_gaussian`.
        """
        self.mesh_data.smooth_gaussian(smoothing_radius, **kwargs)
        if self.has_randoms:
            if check:
                mask_nonzero = self.mesh_randoms.value > 0.
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2: raise ValueError('Very few randoms!')
            self.mesh_randoms.smooth_gaussian(smoothing_radius, **kwargs)
            sum_data, sum_randoms = np.sum(self.mesh_data.value), np.sum(self.mesh_randoms.value)
            alpha = sum_data * 1. / sum_randoms
            self.mesh_delta = self.mesh_data - alpha * self.mesh_randoms
            threshold = ran_min * sum_randoms / self._size_randoms
            if check:
                mean_nran_per_cell = self.mesh_randoms.value[mask_nonzero].mean()
                std_nran_per_cell = self.mesh_randoms.value[mask_nonzero].std(ddof=1)
                self.log_info('Mean smoothed random density in non-empty cells is {:.4f} (std = {:.4f}), threshold is (ran_min * mean weight) = {:.4f}.'.format(mean_nran_per_cell, std_nran_per_cell, threshold))
            mask = self.mesh_randoms > threshold
            if check:
                frac_nonzero_masked = 1. - mask.sum() / nnonzero
                del mask_nonzero
                if frac_nonzero_masked > 0.1:
                    self.log_warning('Masking a large fraction {:.4f} of non-empty cells. You should probably increase the number of randoms.'.format(frac_nonzero_masked))
                else:
                    self.log_info('Masking a fraction {:.4f} of non-empty cells.'.format(frac_nonzero_masked))
            self.mesh_delta[mask] /= (self.bias * alpha * self.mesh_randoms[mask])
            self.mesh_delta[~mask] = 0.
        else:
            self.mesh_delta = self.mesh_data / np.mean(self.mesh_data) - 1.
            self.mesh_delta /= self.bias
        del self.mesh_data
        del self.mesh_randoms

    def run(self, *args, **kwargs):
        """Run reconstruction; to be implemented in your algorithm."""
        raise NotImplementedError('Implement method "run" in your "{}"-inherited algorithm'.format(self.__class__.___name__))

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
        if self.wrap: positions = self.info.wrap(positions)  # wrap here for local los
        for iaxis, psi in enumerate(self.mesh_psi):
            shifts[:, iaxis] = psi.read_cic(positions, wrap=False)  # already wrapped if required
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

    def read_shifted_positions(self, positions, field='disp+rsd'):
        """
        Read shifted positions i.e. the difference ``positions - self.read_shifts(positions, field=field)``.
        Output (and input) positions are wrapped if :attr:`wrap`.

        Parameters
        ----------
        positions : array of shape (N, 3)
            Cartesian positions.

        field : string, default='disp+rsd'
            Apply either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        Returns
        -------
        positions : array of shape (N, 3)
            Shifted positions.
        """
        shifts = self.read_shifts(positions, field=field)
        positions = positions - shifts
        if self.wrap: positions = self.info.wrap(positions)
        return positions


def _make_property(name):

    @property
    def func(self):
        return getattr(self.info, name)

    return func


for name in ['boxsize', 'boxcenter', 'nmesh', 'offset', 'cellsize']:
    setattr(BaseReconstruction, name, _make_property(name))
