"""
Routines to estimate reconstruction efficiency:

  - :class:`MeshFFTCorrelation`: correlation
  - :class:`MeshFFTTransfer`: transfer
  - :class:`MeshFFTPropagator`: propagator

This requires the following packages:

  - pmesh
  - pypower, see https://github.com/adematti/pypower

"""
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline
from pypower import MeshFFTPower, CatalogMesh, ParticleMesh, ArrayMesh, PowerSpectrumWedges

from .utils import BaseClass
from . import utils


class BasePowerRatio(BaseClass):
    """
    Base template class to compute power ratios.
    Specific statistic should extend this class.
    """
    _powers = ['num', 'denom']
    _attrs = []

    @property
    def ratio(self):
        """Power spectrum ratio."""
        return self.num.power.real/self.denom.power.real

    def __call__(self, k=None, mu=None):
        r"""
        Return :attr:`ratio`, optionally performing linear interpolation over :math:`k` and :math:`\mu`.

        Parameters
        ----------
        k : float, array, default=None
            :math:`k` where to interpolate the power spectrum.
            Values outside :attr:`kavg` are set to the first/last power value;
            outside :attr:`edges[0]` to nan.
            Defaults to :attr:`kavg`.

        mu : float, array, default=None
            :math:`\mu` where to interpolate the power spectrum.
            Defaults to :attr:`muavg`.

        Returns
        -------
        toret : array
            (Optionally interpolated) power spectrum ratio.
        """
        tmp = self.ratio
        if k is None and mu is None:
            return tmp
        kavg, muavg = self.kavg, self.muavg
        if k is None: k = kavg
        if mu is None: mu = muavg
        mask_finite_k, mask_finite_mu = ~np.isnan(kavg), ~np.isnan(muavg)
        kavg, muavg, tmp = kavg[mask_finite_k], muavg[mask_finite_mu], tmp[np.ix_(mask_finite_k, mask_finite_mu)]
        k, mu = np.asarray(k), np.asarray(mu)
        isscalar = k.ndim == 0 or mu.ndim == 0
        k, mu = np.atleast_1d(k), np.atleast_1d(mu)
        toret = np.nan * np.zeros((k.size, mu.size), dtype=tmp.dtype)
        mask_k = (k >= self.edges[0][0]) & (k <= self.edges[0][-1])
        mask_mu = (mu >= self.edges[1][0]) & (mu <= self.edges[1][-1])
        if mask_k.any() and mask_mu.any():
            if muavg.size == 1:
                interp = lambda array: UnivariateSpline(kavg, array, k=1, ext='const')(k[mask_k])[:, None]
            else:
                interp = lambda array: RectBivariateSpline(kavg, muavg, array, kx=1, ky=1, s=0)(k[mask_k], mu[mask_mu], grid=True)
            toret[np.ix_(mask_k, mask_mu)] = interp(tmp)
        if isscalar:
            return toret.ravel()
        return toret

    def __copy__(self):
        new = super(BasePowerRatio, self).__copy__()
        for name in self._powers:
            setattr(new, name, getattr(self, name).__copy__())
        return new

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for name in self._powers:
            if hasattr(self, name):
                state[name] = getattr(self, name).__getstate__()
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set this class state."""
        self.__dict__.update(state)
        for name in self._powers:
            if name in state:
                setattr(self, name, PowerSpectrumWedges.from_state(state[name]))

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        self.log_info('Saving {}.'.format(filename))
        utils.mkdir(os.path.dirname(filename))
        np.save(filename, self.__getstate__(), allow_pickle=True)

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new

    def __getitem__(self, slices):
        """Call :meth:`slice`."""
        new = self.copy()
        if isinstance(slices, tuple):
            new.slice(*slices)
        else:
            new.slice(slices)
        return new

    def select(self, *xlims):
        """
        Restrict statistic to provided coordinate limits in place.

        For example:

        .. code-block:: python

            statistic.select((0, 0.3)) # restrict first axis to (0, 0.3)
            statistic.select(None, (0, 0.2)) # restrict second axis to (0, 0.2)

        """
        for name in self._powers:
            getattr(self, name).select(*xlims)

    def slice(self, *slices):
        """
        Slice statistics in place. If slice step is not 1, use :meth:`rebin`.
        For example:

        .. code-block:: python

            statistic.slice(slice(0, 10, 2), slice(0, 6, 3)) # rebin by factor 2 (resp. 3) along axis 0 (resp. 1), up to index 10 (resp. 6)
            statistic[:10:2,:6:3] # same as above, but return new instance.

        """
        for name in self._powers:
            getattr(self, name).slice(*slices)

    def rebin(self, factor=1):
        """
        Rebin statistic, by factor(s) ``factor``.
        A tuple must be provided in case :attr:`ndim` is greater than 1.
        Input factors must divide :attr:`shape`.
        """
        for name in self._powers:
            getattr(self, name).rebin(factor=factor)


def _make_property(name):

    @property
    def func(self):
        return getattr(self.num, name)

    return func

for name in ['edges', 'nmodes', 'modes', 'k', 'mu', 'kavg', 'muavg']:
    setattr(BasePowerRatio, name, _make_property(name))



class MeshFFTCorrelator(BasePowerRatio):
    r"""
    Estimate correlation between two meshes (reconstructed and initial fields), i.e.:

    .. math::

        r(k) = \frac{P_{\mathrm{rec},\mathrm{init}}}{\sqrt{P_{\mathrm{rec}}P_{\mathrm{init}}}}
    """
    _powers = ['num', 'auto_reconstructed', 'auto_initial']

    def __init__(self, mesh_reconstructed, mesh_initial, edges=None, los=None, compensations=None):
        r"""
        Initialize :class:`MeshFFTCorrelation`.

        Parameters
        ----------
        mesh_reconstructed : CatalogMesh, RealField
            Mesh with reconstructed density field.
            If ``RealField``, should be :math:`1 + \delta` or :math:`\bar{n} (1 + \delta)`.

        mesh_initial : CatalogMesh, RealField
            Mesh with initial density field (before structure formation).
            If ``RealField``, should be :math:`1 + \delta` or :math:`\bar{n} (1 + \delta)`.

        edges : tuple, array, default=None
            :math:`k`-edges for :attr:`poles`.
            One can also provide :math:`\mu-edges` (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'step' (if not provided :func:`pypower.fft_power.find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').

        los : array, default=None
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        compensations : list, tuple, default=None
            Compensations to apply to mesh to (optionally) correct for particle-mesh assignment scheme;
            e.g. 'cic-sn' (resp. 'cic') for cic assignment scheme, with (resp. without) interlacing.
            Provide a list (or tuple) of two such strings (for ``mesh_reconstructed`` and `mesh_initial``, respectively).
            Used only if provided ``mesh_reconstructed`` or ``mesh_initial`` are not ``CatalogMesh``.
        """
        if los is None:
            raise ValueError('Provide a box axis or a vector as line-of-sight (no varying line-of-sight accepted)')
        num = MeshFFTPower(mesh_reconstructed, mesh2=mesh_initial, edges=edges, ells=None, los=los, compensations=compensations, shotnoise=0.)
        self.num = num.wedges
        # If compensations is a tuple, the code will anyway use the first element
        self.auto_reconstructed = MeshFFTPower(mesh_reconstructed, edges=edges, ells=None, los=los, compensations=num.compensations).wedges
        self.auto_initial = MeshFFTPower(mesh_initial, edges=edges, ells=None, los=los, compensations=num.compensations[::-1]).wedges

    @property
    def ratio(self):
        """Power spectrum ratio."""
        return self.num.power.real/(self.auto_reconstructed.power.real*self.auto_initial.power.real)**0.5

    def to_propagator(self, growth=1.):
        """
        Return propagator, using computed power spectra.

        Parameters
        ----------
        growth : float, default=1.
            Growth factor (and galaxy bias) to turn initial field to the linearly-evolved galaxy density field at the redshift of interest.

        Returns
        -------
        propagator : MeshFFTPropagator
        """
        new = MeshFFTPropagator.__new__(MeshFFTPropagator)
        new.num = self.num
        new.denom = self.auto_initial
        new.growth = growth
        return new

    def to_transfer(self, growth=1.):
        """
        Return transfer function, using computed power spectra.

        Parameters
        ----------
        growth : float, default=1.
            Growth factor (and galaxy bias) to turn initial field to the linearly-evolved galaxy density field at the redshift of interest.

        Returns
        -------
        transfer : MeshFFTTransfer
        """
        new = MeshFFTTransfer.__new__(MeshFFTTransfer)
        new.num = self.auto_reconstructed
        new.denom = self.auto_initial
        new.growth = growth
        return new


class MeshFFTTransfer(BasePowerRatio):
    r"""
    Estimate transfer function, i.e.:

    .. math::

        t(k) = \frac{P_{\mathrm{rec}}}{G^{2}(z) b^{2}(z) P_{\mathrm{init}}}
    """
    _attrs = ['growth']

    def __init__(self, mesh_reconstructed, mesh_initial, edges=None, los=None, compensations=None, growth=1.):
        """
        Initialize :class:`MeshFFTTransfer`.

        Parameters
        ----------
        mesh_reconstructed : CatalogMesh, RealField
            Mesh with reconstructed density field.

        mesh_initial : CatalogMesh, RealField
            Mesh with initial density field (before structure formation).

        edges : tuple, array, default=None
            :math:`k`-edges for :attr:`poles`.
            One can also provide :math:`\mu-edges` (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'step' (if not provided :func:`pypower.fft_power.find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').

        los : array, default=None
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        compensations : list, tuple, default=None
            Compensations to apply to mesh to (optionally) correct for particle-mesh assignment scheme;
            e.g. 'cic-sn' (resp. 'cic') for cic assignment scheme, with (resp. without) interlacing.
            Provide a list (or tuple) of two such strings (for ``mesh_reconstructed`` and `mesh_initial``, respectively).
            Used only if provided ``mesh_reconstructed`` or ``mesh_initial`` are not ``CatalogMesh``.

        growth : float, default=1.
            Growth factor (and galaxy bias) to turn initial field to the linearly-evolved galaxy density field at the redshift of interest.
        """
        num = MeshFFTPower(mesh_reconstructed, edges=edges, ells=None, los=los, compensations=compensations)
        self.num = num.wedges
        self.denom = MeshFFTPower(mesh_initial, edges=edges, ells=None, los=los, compensations=num.compensations[::-1]).wedges
        self.growth = growth

    @property
    def ratio(self):
        """Power spectrum ratio, normalizing by growth."""
        return (self.num.power.real/self.denom.power.real)**0.5/self.growth


class MeshFFTPropagator(BasePowerRatio):
    r"""
    Estimate propagator, i.e.:

    .. math::

        g(k) = \frac{P_{\mathrm{rec},\mathrm{init}}}{G(z) b(z) P_{\mathrm{init}}}
    """
    _attrs = ['growth']

    def __init__(self, mesh_reconstructed, mesh_initial, edges=None, los=None, compensations=None, growth=1.):
        """
        Initialize :class:`MeshFFTPropagator`.

        Parameters
        ----------
        mesh_reconstructed : CatalogMesh, RealField
            Mesh with reconstructed density field.

        mesh_initial : CatalogMesh, RealField
            Mesh with initial density field (before structure formation).

        edges : tuple, array, default=None
            :math:`k`-edges for :attr:`poles`.
            One can also provide :math:`\mu-edges` (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'step' (if not provided :func:`pypower.fft_power.find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').

        los : array, default=None
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        compensations : list, tuple, default=None
            Compensations to apply to mesh to (optionally) correct for particle-mesh assignment scheme;
            e.g. 'cic-sn' (resp. 'cic') for cic assignment scheme, with (resp. without) interlacing.
            Provide a list (or tuple) of two such strings (for ``mesh_reconstructed`` and `mesh_initial``, respectively).
            Used only if provided ``mesh_reconstructed`` or ``mesh_initial`` are not ``CatalogMesh``.

        growth : float, default=1.
            Growth factor (and galaxy bias) to turn initial field to the linearly-evolved galaxy density field at the redshift of interest.
        """
        num = MeshFFTPower(mesh_reconstructed, mesh2=mesh_initial, edges=edges, ells=None, los=los, compensations=compensations, shotnoise=0.)
        self.num = num.wedges
        self.denom = MeshFFTPower(mesh_initial, edges=edges, ells=None, los=los, compensations=num.compensations[::-1]).wedges
        self.growth = growth

    @property
    def ratio(self):
        """Power spectrum ratio, normalizing by growth."""
        return self.num.power.real/self.denom.power.real/self.growth
