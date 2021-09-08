import os
import ctypes

import numpy as np
from numpy import ctypeslib

from .recon import BaseReconstruction
from .mesh import RealMesh
from . import utils


class MultiGridReconstruction(BaseReconstruction):
    """
    :mod:`ctypes`-based implementation for Martin J. White reconstruction code,
    using full multigrid V-cycle based on damped Jacobi iteration.
    So far we stick to the implementation at https://github.com/martinjameswhite/recon_code.
    """
    _path_lib = os.path.join(utils.lib_dir,'multigrid_{}.so')

    def __init__(self, *args, **kwargs):
        """
        Initialize :class:`MultiGridReconstruction`.
        See :class:`BaseReconstruction` for input parameters.
        """
        super(MultiGridReconstruction,self).__init__(*args,**kwargs)
        self._type_float = self.mesh_data._type_float
        self._lib = ctypes.CDLL(self._path_lib.format(self.mesh_data._precision),mode=ctypes.RTLD_LOCAL)

    def run(self, jacobi_damping_factor=0.4, jacobi_niterations=5, vcycle_niterations=6):
        """
        Run reconstruction, i.e. set displacement potential attr:`mesh_phi` from :attr:`mesh_delta`.
        Default parameter values are the same as in Martin's code.

        Parameters
        ----------
        jacobi_damping_factor : float, default=0.4
            Damping factor for Jacobi iterations.

        jacobi_niterations : int, default=5
            Number of Jacobi iterations.

        vcycle_niterations : int, default=6
            Number of V-cycle calls.
        """
        func = self._lib.fmg
        ndim = 3
        type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int,shape=ndim)
        type_boxsize = ctypeslib.ndpointer(dtype=self._type_float,shape=ndim)
        func.argtypes = (self.mesh_delta._type_mesh,self.mesh_delta._type_mesh,
                        type_nmesh,type_boxsize,type_boxsize,
                        self._type_float,self._type_float,ctypes.c_int,ctypes.c_int)
        self.mesh_phi = self.mesh_delta.zeros_like()
        self.mesh_phi.value.shape = -1
        self.log_info('Computing displacement potential.')
        func(self.mesh_delta.value.ravel(),self.mesh_phi.value,
            self.mesh_delta.nmesh.astype(ctypes.c_int,copy=False),self.mesh_delta.boxsize.astype(self._type_float,copy=False),self.mesh_delta.boxcenter.astype(self._type_float,copy=False),
            self.beta,jacobi_damping_factor,jacobi_niterations,vcycle_niterations)
        self.mesh_phi.value.shape = self.mesh_delta.shape

    def read_shifts(self, positions, with_rsd=True):
        """
        Read Zeldovich displacement at input positions by deriving the computed displacement potential :attr:`mesh_phi` (finite difference scheme).
        See :meth:`BaseReconstruction.read_shifts` for input parameters.
        """
        shifts = self.mesh_phi.read_finite_difference_cic(positions)
        if with_rsd:
            los = positions/utils.distance(positions)[:,None]
            shifts += self.f*np.sum(shifts*los,axis=-1)[:,None]*los
        return shifts
