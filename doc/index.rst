.. title:: pyrecon docs

***********************************
Welcome to pyrecon's documentation!
***********************************

.. toctree::
  :maxdepth: 1
  :caption: User documentation

  user/building
  api/api

.. toctree::
  :maxdepth: 1
  :caption: Developer documentation

  developer/documentation
  developer/tests
  developer/contributing
  developer/changes

.. toctree::
  :hidden:

************
Introduction
************

**pyrecon** is a Python package to perform reconstruction in BAO analyses with various algorithms with MPI.

A typical reconstruction run is (e.g. for MultiGridReconstruction; the same works for other algorithms):

.. code-block:: python

  from pyrecon import MultiGridReconstruction

  # line-of-sight "los" can be local (None, default) or an axis, 'x', 'y', 'z', or a 3-vector
  recon = MultiGridReconstruction(f=0.8, bias=2.0, los=None, nmesh=512, boxsize=1000., boxcenter=2000.)
  recon.assign_data(positions_data, weights_data) # positions_data are a (N, 3) array of Cartesian positions, weights a (N,) array
  # You can skip the following line if you assume uniform selection function (randoms)
  recon.assign_randoms(positions_randoms, weights_randoms)
  recon.set_density_contrast()
  recon.run()
  # If you are using IterativeFFTParticleReconstruction, displacements are to be taken at the reconstructed data real-space positions;
  # In this case, do: positions_rec_data = recon.read_shifted_positions('data')
  positions_rec_data = recon.read_shifted_positions(positions_data)
  # RecSym = remove large scale RSD from randoms
  positions_rec_randoms = recon.read_shifted_positions(positions_randoms)
  # or RecIso
  # positions_rec_randoms = recon.read_shifted_positions(positions_randoms, field='disp')


**************
Code structure
**************

The code structure is the following:

  - recon.py implements the base reconstruction class
  - mesh.py implements mesh utilies
  - utils.py implements various utilities
  - a module for each algorithm
  - metrics.py implements calculation of correlator, propagator and transfer function


Changelog
=========

* :doc:`developer/changes`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
