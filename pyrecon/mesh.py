"""Implementation of :class:`RealMesh` and :class:`ComplexMesh`, along with FFT engines."""

import os
import ctypes
import numbers

import numpy as np
from numpy import ctypeslib
from numpy.lib.mixins import NDArrayOperatorsMixin as NDArrayLike

from . import utils
from .utils import BaseClass, BaseMetaClass


class MeshError(Exception):

    """Exception raised when issue with mesh."""


class SetterProperty(object):
    """
    Attribute setter, runs ``func`` when setting a class attribute.
    Taken from https://stackoverflow.com/questions/17576009/python-class-property-use-setter-but-evade-getter
    """
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)


class BaseMesh(NDArrayLike,BaseClass,metaclass=BaseMetaClass):
    """
    Base implementation for mesh.
    What follows are just methods to make :class:`BaseMesh` behave like a numpy array.
    numpy functions can be applied directly to any instance ``mesh`` through e.g.::

        np.sum(mesh)

    Note
    ----
    To get a deep copy of the mesh (including :attr:`value`), use :meth:`deepcopy`.
    :meth:`copy` will return a shallow copy.

    Attributes
    ----------
    value : array
        Numpy array holding mesh values, or ``None`` when unset.
        Can be set any time using ``mesh.value = newvalue``.

    info : MeshInfo
        Mesh information (boxsize, boxcenter, nmesh, etc.)

    dtype : np.dtype
        Type for :attr:`value` array.

    nthreads : int
        Number of threads to use in mesh calculations.

    attrs : dict
        Dictionary of other attributes.

    boxsize : array
        See :class:`MeshInfo`.

    boxcenter : array
        See :class:`MeshInfo`.

    nmesh : array
        See :class:`MeshInfo`.

    cellsize : array
        See :class:`MeshInfo`.

    ndim : array
        See :class:`MeshInfo`.
    """
    _attrs = ['info', 'dtype', 'nthreads', 'attrs']
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    _path_lib = os.path.join(utils.lib_dir, 'mesh_{}.so')

    def __init__(self, value=None, dtype=None, info=None, nthreads=None, attrs=None, **kwargs):
        """
        Initalize :class:`BaseMesh`.

        Parameters
        ----------
        value : array, default=None
            Numpy array holding mesh values, or ``None`` (can set later through ``mesh.value = value``.

        dtype : string, np.dtype, defaut=None
            Type for :attr:`value` array.
            If ``None``, defaults to ``np.asarray(value).dtype`` if ``value`` is not ``None``, else 'f8'.

        info : MeshInfo, default=None
            Mesh information (boxsize, boxcenter, nmesh, etc.),
            copied and updated with ``kwargs``.

        nthreads : int
            Number of threads to use in mesh calculations.

        attrs : dict
            Dictionary of other attributes.

        kwargs : dict
            Arguments for :class:`MeshInfo`.
        """
        if value is not None:
            value = np.asarray(value)
            if dtype is None:
                dtype = value.dtype
        if info is None:
            self.info = MeshInfo(value=value, **kwargs)
        else:
            self.info = info.copy(**kwargs)
        self.value = None
        self.dtype = np.dtype(dtype if dtype is not None else 'f8')
        self.value = value
        self.set_num_threads(nthreads)
        self.attrs = attrs or {}
        self.fft_engine = None

    @SetterProperty
    def dtype(self, dtype):
        """Called when setting :attr:`dtype`, loading the relevant C-library."""
        self.__dict__['dtype'] = np.dtype(dtype)
        self.value = self.value
        self._lib = ctypes.CDLL(self._path_lib.format(self._precision), mode=ctypes.RTLD_LOCAL)

    @property
    def _precision(self):
        # Return float if float32, double if float64
        return self._type_float.__name__[len('c_'):]

    @property
    def _type_float(self):
        # Return ctypes-type corresponding to numpy-dtype
        # Take care of complex type
        if self.dtype.name.startswith('complex'):
            dtype = np.dtype('f{:d}'.format(self.dtype.itemsize//2))
        else:
            dtype = self.dtype
        return ctypeslib.as_ctypes_type(dtype)

    @property
    def _type_float_mesh(self):
        # Return ctypes-type for numpy array
        return ctypeslib.ndpointer(dtype=self._type_float, shape=self.size, flags='C')

    def set_num_threads(self, nthreads=None):
        """Set number of OpenMP threads."""
        if nthreads is not None:
            func = self._lib.set_num_threads
            func.argtypes = (ctypes.c_int,)
            func(nthreads)

    @property
    def nthreads(self):
        """Number of OpenMP threads."""
        func = self._lib.get_num_threads
        func.restype = ctypes.c_int
        return func()

    def __mul__(self, other):
        r = self.deepcopy(copy_value=False)
        r.value = r.value * other
        return r

    def __imul__(self, other):
        self.value *= other
        return self

    def __div__(self, other):
        r = self.deepcopy(copy_value=False)
        r.value = r.value / other
        return r

    __truediv__ = __div__

    def __rdiv__(self, other):
        r = self.deepcopy()
        r.value = other / r.value
        return r

    __rtruediv__ = __rdiv__

    def __idiv__(self, other):
        self.value /= other
        return self

    __itruediv__ = __idiv__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Taken from https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
        # See also https://github.com/rainwoodman/pmesh/blob/master/pmesh/pm.py
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use BaseMesh instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle BaseMesh objects.
            if not isinstance(x, self._HANDLED_TYPES + (BaseMesh,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value if isinstance(x, BaseMesh) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.value if isinstance(x, BaseMesh) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        def cast(result):
            # booleans, cannot be reasonable BaseMesh objects
            # just return the ndarray
            if result.dtype == '?':
                return result
            # different shape, cannot be reasonable Field objects
            # just return the ndarray
            if result.shape != self.shape:
                return result
            # really only cast when we are using simple +-* **, etc.
            new = self.deepcopy(copy_value=False)
            new.value = result
            return new

        if type(result) is tuple:
            # multiple return values
            return tuple(cast(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return cast(result)

    def __getitem__(self, index):
        return self.value.__getitem__(index)

    def __setitem__(self, index, value):
        return self.value.__setitem__(index, value)

    def __array__(self, dtype=None):
        return self.value

    @property
    def shape(self):
        return tuple(self.nmesh)

    @property
    def size(self):
        return np.prod(self.shape)

    @SetterProperty
    def value(self, value):
        if value is not None:
            if isinstance(value, np.ndarray) and value.size == self.size:
                value.shape = self.shape
                value = value.astype(self.dtype, copy=False, order='C')
            else:
                value_ = value
                value = np.empty(shape=self.shape, dtype=self.dtype, order='C')
                value[...] = value_
        self.__dict__['value'] = value

    def zeros_like(self):
        new = self.deepcopy(copy_value=False)
        new.value = np.zeros(shape=self.shape, dtype=self.dtype, order='C')
        return new

    def empty_like(self, *args, **kwargs):
        new = self.deepcopy(copy_value=False)
        new.value = np.empty(shape=self.shape, dtype=self.dtype, order='C')
        return new

    def __repr__(self):
        info = ['{}={}'.format(name,getattr(self.info,name)) for name in self.info._attrs]
        info += ['dtype={}'.format(self.dtype)]
        return '{}({})'.format(self.__class__.__name__,', '.join(info))

    def _copy_value(self, out=None):
        if out is None: out = np.empty_like(self.value)
        func = self._lib.copy
        func.argtypes = (self._type_float_mesh, self._type_float_mesh, ctypes.c_size_t)
        func.restype = ctypes.c_int
        self.value.shape = out.shape = -1
        flag = func(self.value, out, self.size)
        if (flag != 0):
            raise MeshError('Issue with _copy_value')
        self.value.shape = out.shape = self.shape
        return out

    def deepcopy(self, copy_value=True):
        kwargs = {name:getattr(self,name) for name in self._attrs}
        kwargs['info'] = kwargs['info'].deepcopy()
        new = self.__class__(self._copy_value() if copy_value and self.value is not None else self.value,**kwargs)
        new.fft_engine = self.fft_engine
        return new

    def get_fft_engine(self, engine='numpy', **kwargs):
        """
        Return engine for fast Fourier transform.

        Parameters
        ----------
        engine : string, BaseFFTEngine, default='numpy'
            If string, use 'numpy' or 'fftw' (package pyfftw must be installed);
            else a FFT engine.

        kwargs : dict
            Options for the FFT engines, used if ``engine`` is a FFT engine name (string).
            See :class:`NumpyFFTEngine` and :class:`FFTWEngine`.

        Returns
        -------
        engine : BaseFFTEngine
            FFT engine.
        """
        kwargs.setdefault('nthreads',self.nthreads)
        return get_fft_engine(engine,shape=self.shape,type_real=self.dtype,**kwargs)

    def set_fft_engine(self, engine='numpy', **kwargs):
        """
        Set engine for fast Fourier transform.
        See :meth:`get_fft_engine`.
        """
        self.fft_engine = self.get_fft_engine(engine=engine, **kwargs)


def _make_property(name):

    @property
    def func(self):
        return getattr(self.info,name)

    return func

for name in ['boxsize','boxcenter','nmesh','offset','cellsize','ndim']:
    setattr(BaseMesh,name,_make_property(name))


class MeshInfo(BaseClass):
    """
    Class holding mesh information.

    Attributes
    ----------
    boxsize : array
        Physical size of the box.

    boxcenter : array
        Box center.

    nmesh : array
        Mesh size, i.e. number of mesh nodes along each axis.
    """
    _attrs = ['boxsize', 'boxcenter', 'nmesh']

    def __init__(self, nmesh=None, boxsize=None, boxcenter=None, cellsize=None, value=None, positions=None, boxpad=1.5):
        """
        Initalize :class:`MeshInfo`.

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
            If not ``None``, and mesh size ``nmesh`` is not ``None``, used to set ``boxsize`` as ``nmesh * cellsize``.
            If ``nmesh`` is ``None``, it is set as (the nearest integer(s) to) ``boxsize/cellsize``.

        value : array, default=None
            Only used to get mesh size.

        positions : array of shape (N,3), default=None
            If ``boxsize`` and / or ``boxcenter`` is ``None``, use these positions
            to determine ``boxsize`` and / or ``boxcenter``.

        boxpad : float, default=1.5
            When ``boxsize`` is determined from ``positions``, take ``boxpad`` times the smallest box enclosing ``positions`` as ``boxsize``.
        """
        if value is not None and nmesh is None:
            nmesh = value.shape

        if boxsize is None or boxcenter is None:
            if positions is None:
                raise MeshError('boxsize and boxcenter must be specified if positions are not provided')
            pos_min, pos_max = positions.min(axis=0), positions.max(axis=0)
            delta = np.abs(pos_max - pos_min)
            if boxcenter is None: boxcenter = 0.5 * (pos_min + pos_max)
            if boxsize is None:
                if cellsize is not None and nmesh is not None:
                    boxsize = nmesh * cellsize
                else:
                    boxsize = delta.max() * boxpad
            if (boxsize < delta).any(): raise MeshError('boxsize too small to contain all data')

        if nmesh is None:
            if cellsize is not None:
                nmesh = np.rint(boxsize/cellsize).astype(int)
            else:
                raise MeshError('nmesh (or cellsize) must be specified')

        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.nmesh = nmesh

    @SetterProperty
    def boxsize(self, boxsize):
        # Called when setting :attr:`boxsize`, enforcing array of shape (3,).
        _boxsize = np.empty(self.ndim,dtype='f8',order='C')
        _boxsize[:] = boxsize
        self.__dict__['boxsize'] = _boxsize

    @SetterProperty
    def boxcenter(self, boxcenter):
        # Called when setting :attr:`boxcenter`, enforcing array of shape (3,).
        _boxcenter = np.empty(self.ndim,dtype='f8',order='C')
        _boxcenter[:] = boxcenter
        self.__dict__['boxcenter'] = _boxcenter

    @SetterProperty
    def nmesh(self, nmesh):
        # Called when setting :attr:`nmesh`, enforcing array of shape (3,).
        _nmesh = np.empty(self.ndim,dtype='i8',order='C')
        _nmesh[:] = nmesh
        self.__dict__['nmesh'] = _nmesh

    @property
    def offset(self):
        """Coordinates of the (0,0,0) corner of the box."""
        return self.boxcenter - self.boxsize/2.

    @property
    def cellsize(self):
        "Physical size of mesh cells."
        return self.boxsize/self.nmesh

    @property
    def ndim(self):
        """Number of dimensions: 3."""
        return 3

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)


class RealMesh(BaseMesh):

    """Class holding a 3D real mesh."""

    _path_lib = os.path.join(utils.lib_dir,'mesh_{}.so')

    def __init__(self, value=None, dtype=None, info=None, nthreads=None, attrs=None, **kwargs):
        """
        Initalize :class:`RealMesh`.

        Parameters
        ----------
        value : array, default=None
            Numpy array holding mesh values, or ``None`` (can set later through ``mesh.value = value``.

        dtype : string, np.dtype, defaut=None
            Type for :attr:`value` array. Defaults to 'f8'.

        info : MeshInfo, default=None
            Mesh information (boxsize, boxcenter, nmesh, etc.),
            copied and updated with ``kwargs``.

        nthreads : int
            Number of threads to use in mesh calculations.

        attrs : dict
            Dictionary of other attributes.

        kwargs : dict
            Arguments for :class:`MeshInfo`.
        """
        if dtype is None and (value is None or np.ndim(value) == 0): dtype = 'f8' # accept single float as input
        super(RealMesh, self).__init__(value=value, dtype=dtype, info=info, nthreads=nthreads, attrs=attrs, **kwargs)
        if 'float' not in self.dtype.name:
            raise MeshError('Provide float dtype')

    def coords(self):
        """Return array of coordinates along each axis."""
        toret = []
        for idim,(n,o,d) in enumerate(zip(self.nmesh,self.offset,self.boxsize/self.nmesh)):
            toret.append(o + d*np.arange(n))
        return tuple(toret)

    def assign_cic(self, positions, weights=None):
        """
        Assign (paint) positions to mesh with Cloud-in-Cell scheme.

        Parameters
        ----------
        positions : array of shape (N,3)
            Cartesian positions.

        weights : array of shape (N,), default=None
            Weights; default to 1.
        """
        size = len(positions)
        if weights is None: weights = np.ones_like(positions,shape=size,dtype=self._type_float)
        positions = ((positions - self.boxcenter)/self.boxsize + 0.5)*self.nmesh
        positions = positions.astype(self._type_float,copy=False).ravel(order='C')
        weights = weights.astype(self._type_float,copy=False).ravel(order='C')
        if self.value is None: self.value = 0.
        type_positions = ctypeslib.ndpointer(dtype=self._type_float,shape=positions.size,flags='C')
        type_weights = ctypeslib.ndpointer(dtype=self._type_float,shape=weights.size,flags='C')
        type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int,shape=self.ndim,flags='C')
        func = self._lib.assign_cic
        func.argtypes = (self._type_float_mesh,type_nmesh,type_positions,type_weights,ctypes.c_size_t)
        func.restype = ctypes.c_int
        self.value.shape = -1
        flag = func(self.value,self.nmesh.astype(ctypes.c_int,copy=False),positions,weights,size)
        if (flag != 0):
            raise MeshError('Issue with assign_cic')
        self.value.shape = self.shape

    def read_cic(self, positions):
        """
        Read mesh values interpolated at input positions with Cloud-in-Cell scheme.

        Parameters
        ----------
        positions : array of shape (N,3)
            Cartesian positions.

        Returns
        -------
        values : array of shape (N,)
            Mesh values interpolated at input positions.
        """
        size = len(positions)
        positions = ((positions - self.boxcenter)/self.boxsize + 0.5)*self.nmesh
        positions = positions.astype(self._type_float,copy=False).ravel(order='C')
        values = np.empty_like(positions,shape=size,order='C')
        type_positions = ctypeslib.ndpointer(dtype=self._type_float,shape=positions.size,flags='C')
        type_values = ctypeslib.ndpointer(dtype=self._type_float,shape=values.size,flags='C')
        type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int,shape=self.ndim,flags='C')
        func = self._lib.read_cic
        func.argtypes = (self._type_float_mesh,type_nmesh,type_positions,type_values,ctypes.c_size_t)
        func.restype = ctypes.c_int
        flag = func(self.value.ravel(order='C'),self.nmesh.astype(ctypes.c_int,copy=False),positions,values,size)
        if (flag != 0):
            raise MeshError('Issue with read_cic')
        return values

    def read_finite_difference_cic(self, positions):
        """
        Read derivative (finite difference scheme) of mesh values along each axis interpolated at input positions with Cloud-in-Cell scheme.

        Parameters
        ----------
        positions : array of shape (N,3)
            Cartesian positions.

        Returns
        -------
        values : array of shape (N,)
            Derivative of mesh values interpolated at input positions.
        """
        size = len(positions)
        positions = ((positions - self.boxcenter)/self.boxsize + 0.5)*self.nmesh
        positions = positions.astype(self._type_float,copy=False).ravel(order='C')
        values = np.empty_like(positions,order='C')
        type_positions = ctypeslib.ndpointer(dtype=self._type_float,shape=positions.size,flags='C')
        type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int,shape=self.ndim,flags='C')
        type_boxsize = ctypeslib.ndpointer(dtype=self._type_float,shape=self.ndim,flags='C')
        func = self._lib.read_finite_difference_cic
        func.argtypes = (self._type_float_mesh,type_nmesh,type_boxsize,type_positions,type_positions,ctypes.c_size_t)
        func.restype = ctypes.c_int
        flag = func(self.value.ravel(order='C'),self.nmesh.astype(ctypes.c_int,copy=False),self.boxsize.astype(self._type_float,copy=False),positions,values,size)
        if (flag != 0):
            raise MeshError('Issue with read_finite_difference_cic')
        values.shape = (size,self.ndim)
        return values

    def smooth_gaussian(self, radius, method='fft', nsigmas=2.5, **kwargs):
        """
        Apply Gaussian smoothing to mesh.

        Parameters
        ----------
        radius : array, float
            Smoothing scale (along each axis, or same for all axes).

        method : string, default='fft'
            Perform Gaussian smoothing in real space ('real') or using FFT ('fft').

        nsigmas : float, default=2.5
            If ``method`` is 'real', number of Gaussian sigmas where to stop convolution.

        kwargs : dict
            Optional arguments for :meth:`get_fft_engine`.
        """
        radius_ = np.empty_like(self.boxsize,order='C')
        radius_[:] = radius
        if method == 'fft':
            if kwargs or self.fft_engine is None: self.set_fft_engine(**kwargs)
            valuek = self.to_complex()
            k2 = sum(-0.5*(r*k)**2 for r,k in zip(radius_,utils.broadcast_arrays(*valuek.coords())))
            valuek *= np.exp(k2)
            self.value = valuek.to_real().value
            #func = self._lib.smooth_fft_gaussian
            #func.argtypes = (self._type_float_mesh,type_nmesh,type_boxsize)
            #func(self.value.ravel(order='C'),self.nmesh,radius/self.boxsize)
        else:
            radius = radius_/self.boxsize
            type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int,shape=self.ndim,flags='C')
            type_boxsize = ctypeslib.ndpointer(dtype=self._type_float,shape=self.ndim,flags='C')
            func = self._lib.smooth_gaussian
            func.argtypes = (self._type_float_mesh,type_nmesh,type_boxsize,self._type_float)
            self.value.shape = -1
            func.restype = ctypes.c_int
            flag = func(self.value, self.nmesh.astype(ctypes.c_int,copy=False),radius.astype(self._type_float,copy=False),nsigmas)
            if (flag != 0):
                raise MeshError('Issue with read_finite_difference_cic')
            self.value.shape = self.shape

    def to_complex(self, *args, **kwargs):
        """
        Return :class:`ComplexMesh` computed with fast Fourier transforms.
        See :meth:`get_fft_engine` for arguments.
        """
        if kwargs or self.fft_engine is None: self.set_fft_engine(**kwargs)
        toret = ComplexMesh(self.fft_engine.forward(self.value),info=self.info,nthreads=self.nthreads,hermitian=self.fft_engine.hermitian,attrs=self.attrs)
        toret.fft_engine = self.fft_engine
        return toret

    def prod_sum(self, arrays, exp=1):
        """
        Multiply mesh by ``(arrays[0][:,None,None] + arrays[1][None,:,None] + arrays[2][None,None,:]) ** exp``

        Parameters
        ----------
        arrays : sequence of 3 float arrays
            Arrays to multiply mesh by.

        exp : int, default=1
            Exponent to raise broadcast sum of arrays to.
        """
        if len(arrays) != 3:
            raise MeshError('Provide a sequence of 3 arrays')
        arrays = list(arrays)
        #arrays = np.concatenate(arrays[::-1], axis=0, dtype=self._type_float) # ::-1 for prod_sum
        # dtype keyword for np.concatenate appears in version 1.20.0.
        arrays = np.concatenate([np.asarray(array, dtype=self._type_float) for array in arrays[::-1]], axis=0)
        if arrays.size != sum(self.shape):
            raise MeshError('Length of input arrays must match shape')
        type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int, shape=self.ndim, flags='C')
        type_arrays = ctypeslib.ndpointer(dtype=self._type_float, shape=arrays.size, flags='C')
        func = self._lib.prod_sum
        func.argtypes = (self._type_float_mesh, type_nmesh, type_arrays, ctypes.c_int)
        func.restype = ctypes.c_int
        self.value.shape = -1
        flag = func(self.value, self.nmesh.astype(ctypes.c_int,copy=False), arrays, exp)
        self.value.shape = self.shape
        if (flag != 0):
            raise MeshError('Issue with prod_sum')


class ComplexMesh(BaseMesh):
    """
    Class holding a 3D complex mesh.

    Parameters
    ----------
    hermitian : bool
        Whether mesh has Hermitian symmetry, i.e. is real when Fourier transformed.
        In this case, :attr:`shape` is half :attr:`nmesh` on the last axis.
    """
    _attrs = BaseMesh._attrs + ['hermitian']

    def __init__(self, value=None, dtype=None, info=None, hermitian=True, nthreads=None, attrs=None, **kwargs):
        """
        Initialize :class:`ComplexMesh`.

        Parameters
        ----------
        value : array, default=None
            Numpy array holding mesh values, or ``None`` (can set later through ``mesh.value = value``.

        dtype : string, np.dtype, defaut=None
            Type for :attr:`value` array. Defaults to 'c16'.

        info : MeshInfo, default=None
            Mesh information (boxsize, boxcenter, nmesh, etc.).

        hermitian : bool
            Whether mesh has Hermitian symmetry, i.e. is real when Fourier transformed.
            In this case, :attr:`shape` is half :attr:`nmesh` on the last axis.

        nthreads : int
            Number of threads to use in mesh calculations.

        attrs : dict
            Dictionary of other attributes.

        kwargs : dict
            Arguments for :class:`MeshInfo`.
        """
        self.hermitian = hermitian
        if dtype is None and (value is None or np.ndim(value) == 0): dtype = 'c16' # accept single float as input
        super(ComplexMesh, self).__init__(value=value, dtype=dtype, info=info, nthreads=nthreads, attrs=attrs, **kwargs)
        if 'complex' not in self.dtype.name:
            raise MeshError('Provide complex dtype')

    def _copy_value(self, out=None):
        if out is None: out = np.empty_like(self.value)
        func = self._lib.copy
        type_mesh = ctypeslib.ndpointer(dtype=self._type_float, shape=2*self.size, flags='C')
        func.argtypes = (type_mesh, type_mesh, ctypes.c_size_t)
        func.restype = ctypes.c_int
        value_view = self.value.view(dtype=self._type_float)
        out_view = out.view(dtype=self._type_float)
        value_view.shape = out_view.shape = -1
        flag = func(value_view, out_view, 2*self.size)
        out = out_view.view(dtype=self.dtype)
        out.shape = self.shape
        if (flag != 0):
            raise MeshError('Issue with _copy_value')
        return out

    @property
    def shape(self):
        if self.hermitian:
            return tuple(self.nmesh[:-1]) + (self.nmesh[-1]//2 + 1,)
        return tuple(self.nmesh)

    @property
    def fundamental_freq(self):
        """Fundamental frequency of the mesh along each axis."""
        return 2.*np.pi/self.boxsize

    def coords(self):
        """Return array of frequency (wavenumbers) along each axis."""
        toret = []
        for idim,(n,d) in enumerate(zip(self.nmesh, self.boxsize/self.nmesh)):
            if (not self.hermitian) or (idim < self.ndim - 1):
                toret.append(2*np.pi*np.fft.fftfreq(n,d=d))
            else:
                toret.append(2*np.pi*np.fft.rfftfreq(n,d=d))
        return tuple(toret)

    def get_fft_engine(self, engine='numpy', **kwargs):
        """Same as :meth:`RealMesh.get_fft_engine`."""
        kwargs.setdefault('nthreads', self.nthreads)
        return get_fft_engine(engine, shape=self.nmesh, type_complex=self.dtype, hermitian=self.hermitian, **kwargs)

    def to_real(self, *args, **kwargs):
        """
        Return :class:`RealMesh` computed with fast Fourier transforms.
        See :meth:`get_fft_engine` for arguments.
        Raises a :class:`MeshError` if FFT engine has not same Hermitian symmetry.
        """
        if kwargs or self.fft_engine is None: self.set_fft_engine(**kwargs)
        if self.fft_engine.hermitian != self.hermitian:
            raise MeshError('ComplexMesh has hermitian = {} but provided FFT engine has hermitian = {}'.format(self.hermitian,self.fft_engine.hermitian))
        value = self.value
        kwargs = {}
        if isinstance(self.fft_engine, FFTWEngine):
            if self.fft_engine.hermitian: # input destroyed only when hermitian
                value = self._copy_value()
            kwargs = {'destroy_input':True}
        toret = RealMesh(self.fft_engine.backward(value, **kwargs).real, info=self.info, nthreads=self.nthreads, attrs=self.attrs)
        toret.fft_engine = self.fft_engine
        return toret

    def prod_sum(self, arrays, exp=1):
        """
        Multiply mesh by ``(arrays[0][:,None,None] + arrays[1][None,:,None] + arrays[2][None,None,:]) ** exp``

        Parameters
        ----------
        arrays : sequence of 3 float arrays
            Arrays to multiply mesh by.

        exp : int, default=1
            Exponent to raise broadcast sum of arrays to.
        """
        if len(arrays) != 3:
            raise MeshError('Provide a sequence of 3 arrays')
        arrays = list(arrays)
        arrays[-1] = np.repeat(arrays[-1], 2)
        #arrays = np.concatenate(arrays[::-1], axis=0, dtype=self._type_float) # ::-1 for prod_sum
        # dtype keyword for np.concatenate appears in version 1.20.0.
        arrays = np.concatenate([np.asarray(array, dtype=self._type_float) for array in arrays[::-1]], axis=0)
        if arrays.size != sum(self.shape) + self.shape[-1]:
            raise MeshError('Length of input arrays must match shape')
        shape = np.asarray(self.shape, dtype=ctypes.c_int)
        shape[-1] *= 2
        type_mesh = ctypeslib.ndpointer(dtype=self._type_float, shape=np.prod(shape), flags='C')
        type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int, shape=self.ndim, flags='C')
        type_arrays = ctypeslib.ndpointer(dtype=self._type_float, shape=arrays.size, flags='C')
        func = self._lib.prod_sum
        func.argtypes = (type_mesh, type_nmesh, type_arrays, ctypes.c_int)
        func.restype = ctypes.c_int
        #value = np.array(self.value)
        #print(value.shape, value.size)
        #value.shape = (value.size,)
        self.value.shape = -1
        value_view = self.value.view(dtype=self._type_float)
        flag = func(value_view, shape, arrays, exp)
        self.value = value_view.view(dtype=self.dtype)
        if (flag != 0):
            raise MeshError('Issue with prod_sum')

class BaseFFTEngine(object):
    """
    Base engine for fast Fourier transforms.
    FFT engines should extend this class, by (at least) implementing:

    - :meth:`forward`
    - :meth:`backward`

    Attributes
    ----------
    shape : tuple
        Shape of array (in real-space, i.e. not accounting for Hermitian symmetry) to transform.

    nthreads : int
        Number of threads.

    type_real : np.dtype
        Type for real values.

    type_complex : np.dtype
        Type for complex values. Twice larger than :attr:`type_float`.

    hermitian : bool
        Whether complex array has Hermitian symmetry, i.e. is real when Fourier transformed.
    """
    def __init__(self, shape, nthreads=None, type_complex=None, type_real=None, hermitian=True):
        """
        Initialize FFT engine.
        Default types are 'c16' for :attr:`type_complex` and 'f8' for :attr:`type_float`.

        Parameters
        ----------
        shape : list, tuple
            Array shape.

        nthreads : int, default=None
            Number of threads.

        type_complex : string, np.dtype, default=None
            Type for complex values.
            If not provided, use ``type_real`` instead.

        type_real : string, np.dtype, default=None
            Type for real values.
            If not provided, use ``type_complex`` instead.
        """
        if nthreads is None:
            self.nthreads = int(os.environ.get('OMP_NUM_THREADS',1))
        else:
            self.nthreads = nthreads
        self.shape = tuple(shape)
        if type_complex is not None:
            self.type_complex = np.dtype(type_complex)
            itemsize = np.dtype(self.type_complex).itemsize
            self.type_real = np.dtype('f{:d}'.format(itemsize//2))
        else:
            if type_real is None: type_real = 'f8'
            self.type_real = np.dtype(type_real)
            itemsize = np.dtype(self.type_real).itemsize
            self.type_complex = np.dtype('c{:d}'.format(itemsize*2))
        self.hermitian = hermitian

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Size of array (in real-space, i.e. not accounting for Hermitian symmetry) to transform."""
        return np.prod(self.shape)

    @property
    def hshape(self):
        """Shape in Fourier-space, accounting for Hermitian symmetry."""
        if self.hermitian:
            return self.shape[:-1] + (self.shape[-1]//2 + 1,)
        return tuple(self.shape)

    def forward(self, fun):
        """Return forward transform of ``fun``."""
        raise NotImplementedError('Implement "forward" method in your "BaseFFTEngine"-inherited FFT engine.')

    def backward(self, fun):
        """Return backward transform of ``fun``."""
        raise NotImplementedError('Implement "backward" method in your "BaseFFTEngine"-inherited FFT engine.')


class NumpyFFTEngine(BaseFFTEngine):

    """FFT engine based on :mod:`numpy.fft`."""

    def forward(self, fun):
        """Return forward transform of ``fun``."""
        if self.hermitian:
            return np.fft.rfftn(fun).astype(self.type_complex,copy=False)
        return np.fft.fftn(fun).astype(self.type_complex,copy=False)

    def backward(self, fun):
        """Return backward transform of ``fun``."""
        if self.hermitian:
            return np.fft.irfftn(fun).astype(self.type_real,copy=False)
        return np.fft.ifftn(fun).astype(self.type_complex,copy=False)


try: import pyfftw
except ImportError: pyfftw = None


class FFTWEngine(BaseFFTEngine):

    """FFT engine based on :mod:`pyfftw`."""

    def __init__(self, shape, nthreads=None, wisdom=None, plan='measure', **kwargs):
        """
        Initialize :mod:`pyfftw` engine.

        Note
        ----
        :class:`pyfftw.FFTW` internally stores :attr:`pyfftw.FFTW._input_array` and :attr:`pyfftw.FFTW._output_array`,
        which is a waste of memory if one does not want to save them.
        e.g. performing ``engine.backward(engine.forward(array))`` would take as much as 3 times
        (2 for the forward transform, and 1 output array in the backward transform) the memory footprint of ``array``.
        As no access is provided to :attr:`pyfftw.FFTW._input_array` and :attr:`pyfftw.FFTW._output_array` attributes,
        we choose to destroy and rebuild :class:`pyfftw.FFTW` for each transform, thereby allowing Python to destroy
        undesired arrays, at a relatively modest overhead (~ 0.5 s).

        Parameters
        ----------
        shape : list, tuple
            Array shape.

        nthreads : int, default=None
            Number of threads.

        wisdom : string, tuple, default=None
            :mod:`pyfftw` wisdom, used to accelerate further FFTs.
            If a string, should be a path to the save FFT wisdom (with :func:`numpy.save`).
            If a tuple, directly corresponds to the wisdom.

        plan : string, default='measure'
            Choices are ['estimate', 'measure', 'patient', 'exhaustive'].
            The increasing amount of effort spent during the planning stage to create the fastest possible transform.
            Usually 'measure' is a good compromise.

        kwargs : dict
            Optional arguments for :class:`BaseFFTEngine`.
        """
        if pyfftw is None:
            raise NotImplementedError('Install pyfftw to use {}'.format(self.__class__.__name__))
        super(FFTWEngine,self).__init__(shape,nthreads=nthreads,**kwargs)
        plan = plan.lower()
        allowed_plans = ['estimate', 'measure', 'patient', 'exhaustive']
        if plan not in allowed_plans:
            raise MeshError('Plan {} unknown'.format(plan))
        plan = 'FFTW_{}'.format(plan.upper())

        if isinstance(wisdom, str):
            wisdom = tuple(np.load(wisdom))
        if wisdom is not None:
            pyfftw.import_wisdom(wisdom)
        else:
            pyfftw.forget_wisdom()
        if self.hermitian:
            fftw_f = pyfftw.empty_aligned(self.shape,dtype=self.type_real,order='C')
        else:
            fftw_f = pyfftw.empty_aligned(self.shape,dtype=self.type_complex,order='C')
        fftw_fk = pyfftw.empty_aligned(self.hshape,dtype=self.type_complex,order='C')
        self.flags = (plan,)
        v = pyfftw.FFTW(fftw_f,fftw_fk,axes=range(self.ndim),direction='FFTW_FORWARD',flags=self.flags,threads=self.nthreads)
        self.fftw_forward_object = pyfftw.FFTW(fftw_f,fftw_fk,axes=range(self.ndim),direction='FFTW_FORWARD',flags=self.flags,threads=self.nthreads)
        self.fftw_backward_object = pyfftw.FFTW(fftw_fk,fftw_f,axes=range(self.ndim),direction='FFTW_BACKWARD',flags=self.flags,threads=self.nthreads)
        # We delete these instances to save memory, see note above
        self.fftw_forward_object, self.fftw_backward_object = None, None

    def forward(self, fun):
        """Return forward transform of ``fun``."""
        output_array = pyfftw.empty_aligned(self.hshape,dtype=self.type_complex,order='C')
        #if self.hermitian:
        #    input_array = pyfftw.empty_aligned(self.shape,dtype=self.type_real,order='C')
        #else:
        #    input_array = pyfftw.empty_aligned(self.shape,dtype=self.type_complex,order='C')
        if self.hermitian:
            fun = fun.astype(self.type_real,copy=False)
        else:
            fun = fun.astype(self.type_complex,copy=False)
        if self.fftw_forward_object is None:
            fftw_forward_object = pyfftw.FFTW(fun,output_array,axes=range(self.ndim),direction='FFTW_FORWARD',flags=self.flags,threads=self.nthreads)
            #input_array[...] = fun
            toret = fftw_forward_object(normalise_idft=True)
        else:
            toret = self.fftw_forward_object(input_array=fun,output_array=output_array,normalise_idft=True)
        return toret

    def backward(self, fun, destroy_input=True):
        """Return backward transform of ``fun``; ``destroy_input = True`` to allow destroy ``fun`` (in case dimension > 1 and hermitian)."""
        if destroy_input:
            input_array = fun
        else:
            input_array = pyfftw.empty_aligned(self.hshape,dtype=self.type_complex,order='C')
            input_array[...] = fun
        if self.hermitian:
            output_array = pyfftw.empty_aligned(self.shape,dtype=self.type_real,order='C')
        else:
            output_array = pyfftw.empty_aligned(self.shape,dtype=self.type_complex,order='C')
        if self.fftw_backward_object is None:
            #fftw_backward_object = pyfftw.FFTW(fun,output_array,axes=range(self.ndim),direction='FFTW_BACKWARD',flags=self.flags,threads=self.nthreads)
            fftw_backward_object = pyfftw.FFTW(input_array,output_array,axes=range(self.ndim),direction='FFTW_BACKWARD',flags=self.flags,threads=self.nthreads)
            toret = fftw_backward_object(normalise_idft=True)
        else:
            toret = self.fftw_backward_object(input_array=fun,output_array=output_array,normalise_idft=True)
        return toret


def get_fft_engine(engine, *args, **kwargs):
    """
    Return FFT engine.

    Parameters
    ----------
    engine : BaseFFTEngine, string
        FFT engine, or one of ['numpy', 'fftw'].

    args, kwargs : tuple, dict
        Arguments for FFT engine.

    Returns
    -------
    engine : BaseFFTEngine
    """
    if isinstance(engine, str):
        if engine.lower() == 'numpy':
            return NumpyFFTEngine(*args, **kwargs)
        if engine.lower() == 'fftw':
            return FFTWEngine(*args, **kwargs)
        raise ValueError('FFT engine {} is unknown'.format(engine))
    return engine
