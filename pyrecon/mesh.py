"""Implementation of :class:`RealMesh` and :class:`ComplexMesh`, along with FFT engines."""

import os
import ctypes
import numbers

import numpy as np
from numpy import ctypeslib
from numpy.lib.mixins import NDArrayOperatorsMixin as NDArrayLike

from . import utils
from .utils import BaseClass, BaseMetaClass


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
    _attrs = ['info','dtype','nthreads','attrs']
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

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
    def dtype(self, dtype):
        self.__dict__['dtype'] = np.dtype(dtype)
        self.value = self.value

    @SetterProperty
    def value(self, value):
        if value is not None:
            if isinstance(value,np.ndarray):
                value.shape = self.shape
                value = value.astype(self.dtype,copy=False)
            else:
                value_ = value
                value = np.empty(shape=self.shape,dtype=self.dtype)
                value[...] = value_
        self.__dict__['value'] = value

    def zeros_like(self):
        new = self.deepcopy(copy_value=False)
        new.value = np.zeros(shape=self.shape,dtype=self.dtype)
        return new

    def empty_like(self, *args, **kwargs):
        new = self.deepcopy(copy_value=False)
        new.value = np.empty(shape=self.shape,dtype=self.dtype)
        return new

    def __repr__(self):
        info = ['{}={}'.format(name,getattr(self.info,name)) for name in self.info._attrs]
        return '{}({})'.format(self.__class__.__name__,', '.join(info))

    def deepcopy(self, copy_value=True):
        kwargs = {name:getattr(self,name) for name in self._attrs}
        kwargs['info'] = kwargs['info'].deepcopy()
        return self.__class__(self.value.copy() if copy_value and self.value is not None else self.value,**kwargs)


def _make_property(name):

    @property
    def func(self):
        return getattr(self.info,name)

    return func

for name in ['boxsize','boxcenter','nmesh','cellsize','ndim']:
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
    _attrs = ['boxsize','boxcenter','nmesh']

    def __init__(self, value=None, boxsize=None, boxcenter=None, cellsize=None, nmesh=256, positions=None, boxpad=1.5):
        """
        Initalize :class:`MeshInfo`.

        Parameters
        ----------
        value : array, default=None
            Only used to get mesh size.
            If not provided, see ``nmesh``.

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

        nmesh : array, int, default=256
            Mesh size, i.e. number of mesh nodes along each axis.

        positions : array of shape (N,3), default=None
            If ``boxsize`` and / or ``boxcenter`` is ``None``, use these positions
            to determine ``boxsize`` and / or ``boxcenter``.

        boxpad : float, default=1.5
            When ``boxsize`` is determined from ``positions``, take ``boxpad`` times the smallest box enclosing ``positions`` as ``boxsize``.
        """
        if value is not None:
            nmesh = value.shape

        if boxsize is None or boxcenter is None:
            if positions is None:
                raise ValueError('boxsize and boxcenter must be specified if positions are not provided')
            pos_min, pos_max = positions.min(axis=0), positions.max(axis=0)
            delta = abs(pos_max - pos_min)
            if boxcenter is None: boxcenter = 0.5 * (pos_min + pos_max)
            if boxsize is None:
                if cellsize is not None and nmesh is not None:
                    boxsize = nmesh * cellsize
                else:
                    boxsize = delta.max() * boxpad
            if (boxsize < delta).any(): raise ValueError('boxsize too small to contain all data')

        if nmesh is None:
            if cellsize is not None:
                nmesh = np.rint(boxsize/cellsize).astype(int)
            else:
                raise ValueError('nmesh (or cellsize) must be specified')

        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.nmesh = nmesh

    @SetterProperty
    def boxsize(self, boxsize):
        # Called when setting :attr:`boxsize`, enforcing array of shape (3,).
        _boxsize = np.empty(self.ndim,dtype='f8')
        _boxsize[:] = boxsize
        self.__dict__['boxsize'] = _boxsize

    @SetterProperty
    def boxcenter(self, boxcenter):
        # Called when setting :attr:`boxcenter`, enforcing array of shape (3,).
        _boxcenter = np.empty(self.ndim,dtype='f8')
        _boxcenter[:] = boxcenter
        self.__dict__['boxcenter'] = _boxcenter

    @SetterProperty
    def nmesh(self, nmesh):
        # Called when setting :attr:`nmesh`, enforcing array of shape (3,).
        _nmesh = np.empty(self.ndim,dtype='i8')
        _nmesh[:] = nmesh
        self.__dict__['nmesh'] = _nmesh

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

    def __init__(self, value=None, dtype='f8', info=None, nthreads=None, attrs=None, **kwargs):
        """
        Initalize :class:`RealMesh`.

        Parameters
        ----------
        value : array, default=None
            Numpy array holding mesh values, or ``None`` (can set later through ``mesh.value = value``.

        dtype : string, np.dtype, defaut='f8'
            Type for :attr:`value` array.

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
            dtype = value.dtype
        if info is None:
            self.info = MeshInfo(value=value,**kwargs)
        else:
            self.info = info.copy(**kwargs)
        self.value = None
        self.dtype = np.dtype(dtype)
        self.value = value
        self.set_num_threads(nthreads)
        self.attrs = attrs or {}

    @SetterProperty
    def dtype(self, dtype):
        """Called when setting :attr:`dtype`, loading the relevant C-library."""
        self.__dict__['dtype'] = np.dtype(dtype)
        self.value = self.value
        self._lib = ctypes.CDLL(self._path_lib.format(self._precision),mode=ctypes.RTLD_LOCAL)

    @property
    def _precision(self):
        # Return float if float32, double if float64
        return self._type_float.__name__[len('c_'):]

    @property
    def _type_float(self):
        # Return ctypes-type corresponding to numpy-dtype
        return ctypeslib.as_ctypes_type(self.dtype)

    @property
    def _type_mesh(self):
        # Return ctypes-type for numpy array
        return ctypeslib.ndpointer(dtype=self._type_float,shape=np.prod(self.nmesh))

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
        positions = positions.astype(self._type_float,copy=False).ravel()
        weights = weights.astype(self._type_float,copy=False).ravel()
        if self.value is None:
            self.value = np.zeros(shape=self.nmesh,dtype=self._type_float)
        type_positions = ctypeslib.ndpointer(dtype=self._type_float,shape=positions.size)
        type_weights = ctypeslib.ndpointer(dtype=self._type_float,shape=weights.size)
        type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int,shape=self.ndim)
        func = self._lib.assign_cic
        func.argtypes = (self._type_mesh,type_nmesh,type_positions,type_weights,ctypes.c_size_t)
        self.value.shape = -1
        func(self.value,self.nmesh.astype(ctypes.c_int,copy=False),positions,weights,size)
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
        positions = positions.astype(self._type_float,copy=False).ravel()
        values = np.empty_like(positions,shape=size)
        type_positions = ctypeslib.ndpointer(dtype=self._type_float,shape=positions.size)
        type_values = ctypeslib.ndpointer(dtype=self._type_float,shape=values.size)
        type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int,shape=self.ndim)
        func = self._lib.read_cic
        func.argtypes = (self._type_mesh,type_nmesh,type_positions,type_values,ctypes.c_size_t)
        func(self.value.ravel(),self.nmesh.astype(ctypes.c_int,copy=False),positions,values,size)
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
        positions = positions.astype(self._type_float,copy=False).ravel()
        values = np.empty_like(positions)
        type_positions = ctypeslib.ndpointer(dtype=self._type_float,shape=positions.size)
        type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int,shape=self.ndim)
        type_boxsize = ctypeslib.ndpointer(dtype=self._type_float,shape=self.ndim)
        func = self._lib.read_finite_difference_cic
        func.argtypes = (self._type_mesh,type_nmesh,type_boxsize,type_positions,type_positions,ctypes.c_size_t)
        func(self.value.ravel(),self.nmesh.astype(ctypes.c_int,copy=False),self.boxsize.astype(self._type_float,copy=False),positions,values,size)
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
        radius_ = np.empty_like(self.boxsize)
        radius_[:] = radius
        if method == 'fft':
            engine = self.get_fft_engine(**kwargs)
            valuek = self.to_complex(engine=engine)
            k2 = sum(-0.5*(r*k)**2 for r,k in zip(radius_,utils.broadcast_arrays(*valuek.freq())))
            valuek *= np.exp(k2)
            self.value = valuek.to_real(engine=engine).value
            #func = self._lib.smooth_fft_gaussian
            #func.argtypes = (self._type_mesh,type_nmesh,type_boxsize)
            #func(self.value.ravel(),self.nmesh,radius/self.boxsize)
        else:
            radius = radius_/self.boxsize
            type_nmesh = ctypeslib.ndpointer(dtype=ctypes.c_int,shape=self.ndim)
            type_boxsize = ctypeslib.ndpointer(dtype=self._type_float,shape=self.ndim)
            func = self._lib.smooth_gaussian
            func.argtypes = (self._type_mesh,type_nmesh,type_boxsize,self._type_float)
            self.value.shape = -1
            func(self.value,self.nmesh.astype(ctypes.c_int,copy=False),radius.astype(self._type_float,copy=False),nsigmas)
            self.value.shape = self.shape

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

    def to_complex(self, *args, **kwargs):
        """
        Return :class:`ComplexMesh` computed with fast Fourier transforms.
        See :meth:`get_fft_engine` for arguments.
        """
        engine = self.get_fft_engine(*args,**kwargs)
        return ComplexMesh(engine.forward(self.value),info=self.info,nthreads=self.nthreads,hermitian=engine.hermitian,attrs=self.attrs)


class ComplexMesh(BaseMesh):
    """
    Class holding a 3D complex mesh.

    Parameters
    ----------
    hermitian : bool
        Whether mesh has Hermitian symmetry, i.e. is real when Fourier transformed.
        In this case, :attr:`shape` is half :attr:`nmesh` on the last axis.
    """
    _attrs = BaseMesh._attrs + ['hermitian','nthreads']

    def __init__(self, value=None,  dtype='c16', info=None, nthreads=None, hermitian=True, attrs=None):
        """
        Initialize :class:`ComplexMesh`.

        Parameters
        ----------
        value : array, default=None
            Numpy array holding mesh values, or ``None`` (can set later through ``mesh.value = value``.

        dtype : string, np.dtype, defaut='c16'
            Type for :attr:`value` array.

        info : MeshInfo, default=None
            Mesh information (boxsize, boxcenter, nmesh, etc.).

        nthreads : int
            Number of threads to use in mesh calculations.

        hermitian : bool
            Whether mesh has Hermitian symmetry, i.e. is real when Fourier transformed.
            In this case, :attr:`shape` is half :attr:`nmesh` on the last axis.

        attrs : dict
            Dictionary of other attributes.
        """
        self.info = info
        self.hermitian = hermitian
        if value is not None:
            dtype = value.dtype
        self.value = None
        self.dtype = np.dtype(dtype)
        self.value = value
        self.nthreads = nthreads
        self.attrs = attrs or {}

    @property
    def shape(self):
        if self.hermitian:
            return tuple(self.nmesh[:-1]) + (self.nmesh[-1]//2 + 1,)
        return tuple(self.nmesh)

    @property
    def fundamental_freq(self):
        """Fundamental frequency of the mesh along each axis."""
        return 2.*np.pi/self.boxsize

    def freq(self):
        """Return array of frequency (wavenumbers) along each axis."""
        toret = []
        for idim,(n,d) in enumerate(zip(self.nmesh,self.boxsize/self.nmesh)):
            if (not self.hermitian) or (idim < self.ndim - 1):
                toret.append(2*np.pi*np.fft.fftfreq(n,d=d))
            else:
                toret.append(2*np.pi*np.fft.rfftfreq(n,d=d))
        return tuple(toret)

    def get_fft_engine(self, engine='numpy', **kwargs):
        """Same as :meth:`RealMesh.get_fft_engine`."""
        kwargs.setdefault('nthreads',self.nthreads)
        return get_fft_engine(engine,shape=self.nmesh,type_complex=self.dtype,hermitian=self.hermitian,**kwargs)

    def to_real(self, *args, **kwargs):
        """
        Return :class:`RealMesh` computed with fast Fourier transforms.
        See :meth:`get_fft_engine` for arguments.
        Raises a :class:`ValueError` if FFT engine has not same Hermitian symmetry.
        """
        engine = self.get_fft_engine(*args,**kwargs)
        if engine.hermitian != self.hermitian:
            raise ValueError('ComplexMesh has hermitian = {} but provided FFT engine has hermitian = {}'.format(self.hermitian,engine.hermitian))
        return RealMesh(engine.backward(self.value).real,info=self.info,nthreads=self.nthreads,attrs=self.attrs)


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

        nthreads : int
            Number of threads.

        type_complex : string, np.dtype, default=None
            Type for complex values.
            If not provided, use ``type_real`` instead.

        type_real : string, np.dtype, default=None
            Type for real values.
            If not provided, use ``type_complex`` instead.
        """
        if nthreads is not None:
            os.environ['OMP_NUM_THREADS'] = str(nthreads)
        self.nthreads = int(os.environ.get('OMP_NUM_THREADS',1))
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


try:
    import pyfftw
    HAVE_PYFFTW = True
except ImportError:
    HAVE_PYFFTW = False


if HAVE_PYFFTW:

    class FFTWEngine(BaseFFTEngine):

        """FFT engine based on :mod:`pyfftw`."""

        def __init__(self, shape, nthreads=None, wisdom=None, **kwargs):
            """
            Initialize :mod:`pyfftw` engine.

            Parameters
            ----------
            shape : list, tuple
                Array shape.

            nthreads : int
                Number of threads.

            wisdom : string, tuple
                :mod:`pyfftw` wisdom, used to accelerate further FFTs.
                If a string, should be a path to the save FFT wisdom (with :func:`numpy.save`).
                If a tuple, directly corresponds to the wisdom.

            kwargs : dict
                Optional arguments for :class:`BaseFFTEngine`.
            """
            super(FFTWEngine,self).__init__(shape,**kwargs)

            if isinstance(wisdom, str):
                wisdom = tuple(np.load(wisdom))
            if wisdom is not None:
                pyfftw.import_wisdom(wisdom)
            else:
                pyfftw.forget_wisdom()
            if self.hermitian:
                fftw_f = pyfftw.empty_aligned(self.shape,dtype=self.type_real)
            else:
                fftw_f = pyfftw.empty_aligned(self.shape,dtype=self.type_complex)
            fftw_fk = pyfftw.empty_aligned(self.hshape,dtype=self.type_complex)
            self.fftw_forward_object = pyfftw.FFTW(fftw_f,fftw_fk,axes=range(self.ndim),direction='FFTW_FORWARD',threads=self.nthreads)
            self.fftw_backward_object = pyfftw.FFTW(fftw_fk,fftw_f,axes=range(self.ndim),direction='FFTW_BACKWARD',threads=self.nthreads)

        def forward(self, fun):
            """Return forward transform of ``fun``."""
            output_array = pyfftw.empty_aligned(self.hshape,dtype=self.type_complex)
            if self.hermitian:
                fun = fun.astype(self.type_real,copy=False)
            else:
                fun = fun.astype(self.type_complex,copy=False)
            return self.fftw_forward_object(input_array=fun,output_array=output_array,normalise_idft=True)

        def backward(self, fun):
            """Return backward transform of ``fun``, which is destroyed."""
            if self.hermitian:
                output_array = pyfftw.empty_aligned(self.shape,dtype=self.type_real)
            else:
                output_array = pyfftw.empty_aligned(self.shape,dtype=self.type_complex)
            return self.fftw_backward_object(input_array=fun,output_array=output_array,normalise_idft=True)


def get_fft_engine(engine, *args, **kwargs):
    """
    Return FFT engine.

    Parameters
    ----------
    engine : string
        Engine name (so far, 'numpy' or 'fftw').

    args, kwargs : tuple, dict
        Arguments for FFT engine.
        See :class:`NumpyFFTEngine` and :class:`FFTWEngine`.

    Returns
    -------
    engine : BaseFFTEngine
    """
    if engine == 'numpy':
        return NumpyFFTEngine(*args, **kwargs)
    if engine == 'fftw':
        return FFTWEngine(*args, **kwargs)
    if isinstance(engine,BaseFFTEngine):
        return engine
    raise ValueError('FFT engine {} is unknown'.format(engine))
