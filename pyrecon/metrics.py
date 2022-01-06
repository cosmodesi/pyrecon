"""
Routines to estimate reconstruction efficiency:

  - :class:`MeshFFTCorrelation`: correlation
  - :class:`MeshFFTTransfer`: transfer
  - :class:`MeshFFTPropagator`: propagator

This requires the following packages:

  - pmesh
  - pypower, see https://github.com/adematti/pypower

"""

import numpy as np

from mpi4py import MPI

from pmesh.pm import ParticleMesh
from pypower import MeshFFTPower, CatalogMesh, ArrayMesh, WedgePowerSpectrum

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

    @property
    def edges(self):
        """Edges used to bin power spectrum measurements."""
        return self.num.edges

    def _mu_index(self, mu=None):
        if mu is not None:
            return np.digitize(mu, self.edges[1], right=False) - 1
        return Ellipsis

    def k(self, mu=None):
        """Wavenumbers."""
        return self.num.k[:,self._mu_index(mu)]

    def mu(self, mu=None):
        """Cosine angle to line-of-sight of shape :attr:`shape` = (nk, nmu)."""
        return self.num.mu[:,self._mu_index(mu)]

    def __call__(self, mu=None):
        r"""Return :attr:`ratio`, restricted to the bin(s) corresponding to input :math:`\mu` if not ``None``."""
        return self.ratio[:,self._mu_index(mu)]

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
                setattr(self, name, WedgePowerSpectrum.from_state(state[name]))

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        self.log_info('Saving {}.'.format(filename))
        mkdir(os.path.dirname(filename))
        np.save(filename, self.__getstate__(), allow_pickle=True)

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new


class MeshFFTCorrelator(BasePowerRatio):
    r"""
    Estimate correlation between two meshes (reconstructed and initial fields), i.e.:

    .. math::

        r(k) = \frac{P_{\mathrm{rec},\mathrm{init}}}{\sqrt{P_{\mathrm{rec}}P_{\mathrm{init}}}}
    """
    _powers = ['num', 'auto_reconstructed', 'auto_initial']

    def __init__(self, mesh_reconstructed, mesh_initial, edges=None, los=None, compensations=None):
        """
        Initialize :class:`MeshFFTCorrelation`.

        Parameters
        ----------
        mesh_reconstructed : CatalogMesh, RealField
            Mesh with reconstructed density field.

        mesh_initial : CatalogMesh, RealField
            Mesh with initial density field (before structure formation).

        edges : tuple, array, default=None
            :math:`k`-edges for :attr:`poles`.
            One can also provide :math:`\mu-edges` (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.

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
