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
        recon = MyReconstruction(f=0.8,bias=2.0,nmesh=512,boxsize=1000.,boxcenter=2000.)
        recon.assign_data(positions_data,weights_data)
        recon.assign_randoms(positions_randoms,weights_randoms)
        recon.set_density_contrast()
        recon.run()
        positions_rec_data = positions_data - recon.read_shifts(positions_data)
        # RecSym = remove large scale RSD from randoms
        positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms)
        # Or RecIso
        # positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms,with_rsd=False)

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
    """
    def __init__(self, f=0., bias=1., los=None, fft_engine='numpy', fft_wisdom=None, fft_plan=None, **kwargs):
        """
        Initialize :class:`BaseReconstruction`.

        Parameters
        ----------
        f : float
            Growth rate.

        bias : float
            Galaxy bias.

        los : string, array, default=None
            If ``los`` is ``None`` or 'local', use local (varying) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        fft_engine : string, BaseFFTEngine, default='numpy'
            Engine for fast Fourier transforms. See :class:`BaseFFTEngine`.
            We strongly recommend using 'fftw' for multithreaded FFTs.

        fft_wisdom : string, tuple, default=None
            Wisdom for FFTW, if ``fft_engine`` is 'fftw'.

        fft_plan : string, default=None
            Only used for FFTW. Choices are ['estimate', 'measure', 'patient', 'exhaustive'].
            The increasing amount of effort spent during the planning stage to create the fastest possible transform.
            Usually 'measure' is a good compromise.

        kwargs : dict
            Arguments to build :attr:`mesh_data`, :attr:`mesh_randoms` (see :class:`RealMesh`).
        """
        self.set_cosmo(f=f,bias=bias)
        self.mesh_data = RealMesh(**kwargs)
        self.mesh_randoms = RealMesh(**kwargs)
        self.set_los(los)
        self.log_info('Using mesh {}.'.format(self.mesh_data))
        kwargs = {}
        if fft_wisdom is not None: kwargs['wisdom'] = fft_wisdom
        if fft_plan is not None: kwargs['plan'] = fft_plan
        kwargs['hermitian'] = False
        self.mesh_data.set_fft_engine(fft_engine, **kwargs)
        self.mesh_randoms.set_fft_engine(self.mesh_data.fft_engine)

    @property
    def beta(self):
        r""":math:`\beta` parameter, as the ratio of the growth rate to the galaxy bias."""
        return self.f/self.bias

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
        if beta is not None: f = beta*bias
        if f is None: f = self.f
        self.f = f
        self.bias = bias

    def set_los(self, los=None):
        """
        Set line-of-sight.

        Parameters
        ----------
        los : string, array
            If ``los`` is ``None`` or 'local', use local (varying) line-of-sight.
            Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.
        """
        if los in [None, 'local']:
            self.los = None
        elif los in ['x', 'y', 'z']:
            self.los = np.zeros(3, dtype=self.mesh_data.dtype)
            self.los['xyz'.index(los)] = 1.
        else:
            los = np.array(los, dtype=self.mesh_data.dtype)
            self.los = los/utils.distance(los)

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
        self.mesh_data.assign_cic(positions, weights=weights)

    def assign_randoms(self, positions, weights=None):
        """Same as :meth:`assign_data`, but for random objects."""
        self.mesh_randoms.assign_cic(positions, weights=weights)

    @property
    def has_randoms(self):
        return self.mesh_randoms.value is not None

    def set_density_contrast(self, ran_min=0.75, smoothing_radius=15., **kwargs):
        r"""
        Set :math:`\delta` field :attr:`mesh_delta` from data and randoms fields :attr:`mesh_data` and :attr:`mesh_randoms`.
        Eventually we will probably converge on a base method for all reconstructions.

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
            self.mesh_delta = self.mesh_data/np.mean(self.mesh_data) - 1.
            self.mesh_delta /= self.bias
            self.mesh_delta.smooth_gaussian(smoothing_radius,**kwargs)
            return
        # Martin's notes:
        # We remove any points which have too few randoms for a decent
        # density estimate -- this is "fishy", but it tames some of the
        # worst swings due to 1/eps factors. Better would be an interpolation
        # or a pre-smoothing (or many more randoms).
        mask = self.mesh_randoms >= ran_min
        alpha = np.sum(self.mesh_data[mask])/np.sum(self.mesh_randoms[mask])
        # Following two lines are how things are done in original code - does not seem exactly correct so commented out
        #self.mesh_data[(self.mesh_randoms > 0) & (self.mesh_randoms < ran_min)] = 0.
        #alpha = np.sum(self.mesh_data)/np.sum(self.mesh_randoms[mask])
        self.mesh_data[mask] /= alpha*self.mesh_randoms[mask]
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
        self.mesh_delta.smooth_gaussian(smoothing_radius,**kwargs)

    def run(self, *args, **kwargs):
        """Run reconstruction; to be implemented in your algorithm."""
        raise NotImplementedError('Implement method "run" in your "{}"-inherited algorithm'.format(self.__class__.___name__))

    def read_shifts(self, positions, field='disp+rsd'):
        """
        Read displacement at input positions.
        To get reconstructed positions, given reconstruction instance ``recon``:

        .. code-block:: python

            positions_rec_data = positions_data - recon.read_shifts(positions_data)
            # RecSym = remove large scale RSD from randoms
            positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms)
            # Or RecIso
            # positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms, field='disp')

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
        for iaxis,psi in enumerate(self.mesh_psi):
            shifts[:,iaxis] = psi.read_cic(positions)
        if field == 'disp':
            return shifts
        if self.los is None:
            los = positions/utils.distance(positions)[:,None]
        else:
            los = self.los
        rsd = self.f*np.sum(shifts*los,axis=-1)[:,None]*los
        if field == 'rsd':
            return rsd
        # field == 'disp+rsd'
        shifts += rsd
        return shifts
