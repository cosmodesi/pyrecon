.. _user-building:

Building
========

Requirements
------------
Only strict requirement is:

  - numpy

Extra requirements are:

  - `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ (for faster FFTs)
  - fitsio, h5py, astropy to run **pyrecon** as a standalone
  - pypower to evaluate reconstruction metrics (correlation, transfer function and propagator)

pip
---
To install **pyrecon**, simply run::

  python -m pip install git+https://github.com/cosmodesi/pyrecon

To run **pyrecon** as a standalone, a couple of extra dependencies are required (fitsio, h5py, astropy, scipy), which can be installed through::

  python -m pip install git+https://github.com/cosmodesi/pyrecon#egg=pyrecon[extras]

git
---
First::

  git clone https://github.com/cosmodesi/pyrecon.git

To install the code::

  python setup.py install --user

Or in development mode (any change to Python code will take place immediately)::

  python setup.py develop --user