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
        self.log_info('Using mesh {}.'.format(self.mesh_data))
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
        This can be done slab-by-slab (e.g. to reduce memory footprint).

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

    def set_density_contrast(self, **kwargs):
        """
        Set :math:`mesh_delta` field :attr:`mesh_delta` from data and randoms fields :attr:`mesh_data` and :attr:`mesh_randoms`;
        to be re-implemented in your algorithm.
        Eventually we will probably converge on a base method for all reconstructions.
        """
        raise NotImplementedError('Implement method "set_density_contrast" in your "{}"-inherited algorithm'.format(self.__class__.___name__))

    def run(self, *args, **kwargs):
        """Run reconstruction; to be implemented in your algorithm."""
        raise NotImplementedError('Implement method "run" in your "{}"-inherited algorithm'.format(self.__class__.___name__))

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
        raise NotImplementedError('Implement method "read_shifts" in your "{}"-inherited algorithm'.format(self.__class__.___name__))
