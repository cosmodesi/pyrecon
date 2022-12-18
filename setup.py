import os
import sys
from distutils.extension import Extension
from setuptools import setup

import numpy as np

# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))
package_basename = 'pyrecon'

sys.path.insert(0, os.path.join(package_basedir, package_basename))
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
          install_requires=['numpy', 'scipy', 'pmesh'],
          extras_require={'extras': ['mpytools', 'fitsio', 'h5py'], 'metrics': ['pypower @ git+https://github.com/cosmodesi/pypower']},
          ext_modules=[Extension(f'{package_basename}._multigrid', [f'{package_basename}/_multigrid.pyx'],
                       depends=[f'{package_basename}/_multigrid_imp.h', f'{package_basename}/_multigrid_generics.h'],
                       libraries=['m'],
                       include_dirs=['./', np.get_include()])],
          packages=[package_basename],
          scripts=[f'bin/{package_basename}'])
