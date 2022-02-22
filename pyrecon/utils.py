import os
import sys
import time
import logging
import traceback
import functools

import numpy as np

lib_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'lib')


def exception_handler(exc_type, exc_value, exc_traceback):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '='*100
    #log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

    Parameters
    ----------
    level : string, int, default=logging.INFO
        Logging level.

    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.

    filename : string, default=None
        If not ``None`` stream to file name.

    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.

    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level,str):
        level = {'info':logging.INFO,'debug':logging.DEBUG,'warning':logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):

        def format(self, record):
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ' %(asctime)s %(name)-28s %(levelname)-8s %(message)s'
            return super(MyFormatter,self).format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename,mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level,handlers=[handler],**kwargs)
    sys.excepthook = exception_handler


def mkdir(dirname):
    """Try to create ``dirnm`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname) # MPI...
    except OSError:
        return


class BaseMetaClass(type):

    """Meta class to add logging attributes to :class:`BaseClass` derived classes."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        cls.set_logger()
        return cls

    def set_logger(cls):
        """
        Add attributes for logging:

        - logger
        - methods log_debug, log_info, log_warning, log_error, log_critical
        """
        cls.logger = logging.getLogger(cls.__name__)

        def make_logger(level):

            @classmethod
            def logger(cls, *args, **kwargs):
                getattr(cls.logger,level)(*args,**kwargs)

            return logger

        for level in ['debug','info','warning','error','critical']:
            setattr(cls,'log_{}'.format(level),make_logger(level))


class BaseClass(object,metaclass=BaseMetaClass):
    """
    Base class that implements :meth:`copy`.
    To be used throughout this package.
    """
    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self):
        return self.__copy__()


def broadcast_arrays(*arrays):
    """
    Return broadcastable arrays given input 1D arrays.

    Parameters
    ----------
    arrays : 1D arrays
        N 1D arrays of size (n1, n2, ...)

    Returns
    -------
    arrays : ND arrays
        N ND arrays of shape (n1,1,1,1...(N - 1 times)), etc.
    """
    toret = []
    for iaxis,array in enumerate(arrays):
        sl = [None]*len(arrays); sl[iaxis] = slice(None)
        toret.append(array[tuple(sl)])
    return tuple(toret)


def distance(position):
    """Return cartesian distance, taking coordinates along ``position`` last axis."""
    return np.sqrt((position**2).sum(axis=-1))


def cartesian_to_sky(position, wrap=True, degree=True):
    r"""
    Transform Cartesian coordinates into distance, RA, Dec.

    Parameters
    ----------
    position : array of shape (N, 3)
        Position in Cartesian coordinates.

    wrap : bool, default=True
        Whether to wrap RA in :math:`[0, 2 \pi]`.

    degree : bool, default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    Returns
    -------
    dist : array
        Distance.

    ra : array
        Right Ascension.

    dec : array
        Declination.
    """
    dist = distance(position)
    ra = np.arctan2(position[:,1], position[:,0])
    if wrap: ra %= 2.*np.pi
    dec = np.arcsin(position[:,2]/dist)
    conversion = np.pi/180. if degree else 1.
    return dist, ra/conversion, dec/conversion


def sky_to_cartesian(dist, ra, dec, degree=True, dtype=None):
    """
    Transform distance, RA, Dec into Cartesian coordinates.

    Parameters
    ----------
    dist : array of shape (N,)
        Distance.

    ra : array of shape (N,)
        Right Ascension.

    dec : array of shape (N,)
        Declination.

    degree : default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    dtype : numpy.dtype, default=None
        :class:`numpy.dtype` for returned array.

    Returns
    -------
    position : array of shape (N, 3)
        Position in Cartesian coordinates.
    """
    conversion = np.pi/180. if degree else 1.
    position = [None]*3
    cos_dec = np.cos(dec*conversion)
    position[0] = cos_dec*np.cos(ra*conversion)
    position[1] = cos_dec*np.sin(ra*conversion)
    position[2] = np.sin(dec*conversion)
    return (dist*np.asarray(position,dtype=dtype)).T


class DistanceToRedshift(object):

    """Class that holds a conversion distance -> redshift."""

    def __init__(self, distance, zmax=100., nz=2048, interp_order=3):
        """
        Initialize :class:`DistanceToRedshift`.
        Creates an array of redshift -> distance in log(redshift) and instantiates
        a spline interpolator distance -> redshift.

        Parameters
        ----------
        distance : callable
            Callable that provides distance as a function of redshift (array).

        zmax : float, default=100.
            Maximum redshift for redshift <-> distance mapping.

        nz : int, default=2048
            Number of points for redshift <-> distance mapping.

        interp_order : int, default=3
            Interpolation order, e.g. ``1`` for linear interpolation, ``3`` for cubic splines.
        """
        self.distance = distance
        self.zmax = zmax
        self.nz = nz
        zgrid = np.logspace(-8,np.log10(self.zmax),self.nz)
        self.zgrid = np.concatenate([[0.], zgrid])
        self.rgrid = self.distance(self.zgrid)
        from scipy import interpolate
        self.interp = interpolate.UnivariateSpline(self.rgrid,self.zgrid,k=interp_order,s=0)

    def __call__(self, distance):
        """Return (interpolated) redshift at distance ``distance`` (scalar or array)."""
        distance = np.asarray(distance)
        return self.interp(distance).astype(distance.dtype, copy=False)


def _make_array(value, shape, dtype='f8'):
    # Return numpy array filled with value
    toret = np.empty(shape, dtype=dtype)
    toret[...] = value
    return toret


def random_box_positions(boxsize, boxcenter=0., size=None, nbar=None, rng=None, seed=None, dtype=None):
    """
    Return Cartesian positions in a 3D box.

    Parameters
    ----------
    boxsize : array, float
        Physical size of the box.

    boxcenter : array, float, default=0.
        Box center.

    size : float, default=None
        Number of particles. See ``nbar``.

    nbar : float, default=None
        If ``size`` is ``None``, ``size`` is obtained as the nearest integer to ``nbar * volume``
        where ``volume`` is the box volume.

    rng : np.RandomState, default=None
        Random generator, optional.

    seed : int, default=None
        If ``rng`` is ``None``, the random seed.

    dtype : string, np.dtype, defaut=None
        Type output array.

    Returns
    -------
    positions : array of shape (size, 3)
    """
    if rng is None:
        rng = np.random.RandomState(seed=seed)
    ndim = 3
    boxsize = _make_array(boxsize, ndim, dtype=dtype)
    if size is None:
        if nbar is None:
            raise ValueError('Provide either size or nbar')
        size = int(nbar*np.prod(boxsize) + 0.5)
    positions = rng.uniform(0., 1., size=(size, ndim)).astype(dtype)
    boxsize = boxsize.astype(dtype)
    boxcenter = _make_array(boxcenter, ndim, dtype=positions.dtype)
    offset = boxcenter - boxsize/2.
    return positions*boxsize + offset


import time

class MemoryMonitor(object):
    """
    Class that monitors memory usage and clock, useful to check for memory leaks.

    >>> with MemoryMonitor() as mem:
            '''do something'''
            mem()
            '''do something else'''
    """
    def __init__(self, pid=None):
        """
        Initalize :class:`MemoryMonitor` and register current memory usage.

        Parameters
        ----------
        pid : int, default=None
            Process identifier. If ``None``, use the identifier of the current process.
        """
        import psutil
        self.proc = psutil.Process(os.getpid() if pid is None else pid)
        self.mem = self.proc.memory_info().rss / 1e6
        self.time = time.time()
        msg = 'using {:.3f} [Mb]'.format(self.mem)
        print(msg, flush=True)

    def __enter__(self):
        """Enter context."""
        return self

    def __call__(self, log=None):
        """Update memory usage."""
        mem = self.proc.memory_info().rss / 1e6
        t = time.time()
        msg = 'using {:.3f} [Mb] (increase of {:.3f} [Mb]) after {:.3f} [s]'.format(mem,mem-self.mem,t-self.time)
        if log:
            msg = '[{}] {}'.format(log, msg)
        print(msg, flush=True)
        self.mem = mem
        self.time = t

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        self()
