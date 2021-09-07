.. _user-building:

Building
========

Requirements
------------
Only strict requirement is:

  - numpy

Extra requirements are:

  - `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ (for faster FFTs)

git
---
First::

  git clone https://github.com/adematti/pyrecon.git

To install the code::

  python setup.py install --user

Or in development mode (any change to Python code will take place immediately)::

  python setup.py develop --user

pyrecon with Mac OS
--------------------
If you wish to use clang compiler (instead of gcc), you may encounter an error related to ``-fopenmp`` flag.
In this case, you can try to export:

.. code:: bash

  export CC=clang

Before installing pyrecon. This will set clang OpenMP flags for compilation.
Note that with Mac OS gcc can point to clang.
