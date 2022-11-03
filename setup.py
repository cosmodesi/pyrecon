import os
from distutils.extension import Extension
from setuptools import setup

import numpy as np

# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))
package_basename = 'pyrecon'

import _version
version = _version.__version__


if __name__ == '__main__':

    setup(name=package_basename,
          version=version,
          author='cosmodesi',
          author_email='',
          description='Python wrapper for reconstruction codes',
          license='BSD3',
          url='http://github.com/cosmodesi/pyrecon',
          install_requires=['numpy', 'pmesh'],
          extras_require={'extras': ['fitsio', 'h5py', 'astropy'], 'metrics': ['pypower @ git+https://github.com/cosmodesi/pypower']},
          ext_modules=[Extension('pyrecon._multigrid', ['pyrecon/_multigrid.pyx'],
                       depends=['pyrecon/_multigrid_imp.h', 'pyrecon/_multigrid_generics.h'],
                       libraries=['m'],
                       include_dirs=["./", np.get_include()])],
          packages=[package_basename],
          scripts=['bin/pyrecon'])
