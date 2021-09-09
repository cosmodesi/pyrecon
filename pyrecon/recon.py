import numpy as np

from .mesh import RealMesh
from .utils import BaseClass


class BaseReconstruction(BaseClass):
    """
    Base template reconstruction class.
    Reconstruction algorithms should extend this class, by (at least) implementing:

    - :meth:`run`
    - :meth:`read_shifts`

    A standard reconstruction would be:

    .. code-block:: python

        # MyReconstruction is your reconstruction algorithm
        rec = MyReconstruction(f=0.8,bias=2.0,nmesh=512,boxsize=1000.,boxcenter=2000.)
        rec.assign_data(positions_data,weights_data)
        rec.assign_randoms(positions_randoms,weights_randoms)
        rec.set_density_contrast()
        rec.run()
        positions_rec_data = positions_data - rec.read_shifts(positions_data)
        # RecSym = remove large scale RSD from randoms
        positions_rec_randoms = positions_randoms - rec.read_shifts(positions_randoms)
        # Or RecIso
        # positions_rec_randoms = positions_randoms - rec.read_shifts(positions_randoms,with_rsd=False)

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
    """
    def __init__(self, f=0., bias=1., ran_min=0.75, smoothing_radius=15., **kwargs):
        """
        Initialize :class:`BaseReconstruction`.

        Parameters
        ----------
        f : float
            Growth rate.

        bias : float
            Galaxy bias.

        kwargs : dict
            Arguments to build :attr:`mesh_data`, :attr:`mesh_randoms` (see :class:`RealMesh`).
        """
        self.set_cosmo(f=f,bias=bias)
        self.mesh_data = RealMesh(**kwargs)
        self.mesh_randoms = RealMesh(**kwargs)
        self.ran_min = ran_min
        self.smoothing_radius = smoothing_radius

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

    def assign_data(self, positions, weights=None):
        """
        Assign (paint) data to :attr:`mesh_data`.

        Parameters
        ----------
        positions : array of shape (N,3)
            Cartesian positions.

        weights : array of shape (N,), default=None
            Weights; default to 1.
        """
        self.mesh_data.assign_cic(positions,weights=weights)

    def assign_randoms(self, positions, weights=None):
        """Same as :meth:`assign_data`, but for random objects."""
        self.mesh_randoms.assign_cic(positions,weights=weights)

    def set_density_contrast(self, ran_min=0.75, smoothing_radius=15., **kwargs):
        """
        Set :math:`mesh_delta` field :attr:`mesh_delta` from data and randoms fields :attr:`mesh_data` and :attr:`mesh_randoms`.

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
        # Martin's notes:
        # We remove any points which have too few randoms for a decent
        # density estimate -- this is "fishy", but it tames some of the
        # worst swings due to 1/eps factors. Better would be an interpolation
        # or a pre-smoothing (or many more randoms).
        mask = self.mesh_randoms > ran_min
        #print(np.sum((self.mesh_randoms > 0) & (self.mesh_randoms < ran_min)))
        alpha = np.sum(self.mesh_data)/np.sum(self.mesh_randoms)
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
        raise NotImplementedError('Implement "run" method in your "BaseReconstruction"-inherited algorithm')

    def read_shifts(self, positions, with_rsd=True):
        """
        Read Zeldovich displacements; to be implemented in your algorithm.
        To get reconstructed positions, given reconstruction instance ``rec``:

        .. code-block:: python

            positions_rec_data = positions_data - rec.read_shifts(positions_data)
            # RecSym = remove large scale RSD from randoms
            positions_rec_randoms = positions_randoms - rec.read_shifts(positions_randoms)
            # Or RecIso
            # positions_rec_randoms = positions_randoms - rec.read_shifts(positions_randoms,with_rsd=False)

        Parameters
        ----------
        positions : array of shape (N,3)
            Cartesian positions.

        with_rsd : bool, default=True
            Whether (``True``) or not (``False``) to include RSD in the shifts.
        """
        raise NotImplementedError('Implement "read_shifts" method in your "BaseReconstruction"-inherited algorithm')
