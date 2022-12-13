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
  # Instead of boxsize and boxcenter, one can provide a (N, 3) array of Cartesian positions: positions=
  recon = MultiGridReconstruction(f=0.8, bias=2.0, los=None, nmesh=512, boxsize=1000., boxcenter=2000.)
  recon.assign_data(data_positions, data_weights) # data_positions is a (N, 3) array of Cartesian positions, data_weights a (N,) array
  # You can skip the following line if you assume uniform selection function (randoms)
  recon.assign_randoms(randoms_positions, randoms_weights)
  recon.set_density_contrast(smoothing_radius=15.)
  recon.run()
  # A shortcut of the above is:
  # recon = MultiGridReconstruction(f=0.8, bias=2.0, data_positions=data_positions, data_weights=data_weights, randoms_positions=randoms_positions, randoms_weights=randoms_weights, los=None, nmesh=512, boxsize=1000., boxcenter=2000.)
  # If you are using IterativeFFTParticleReconstruction, displacements are to be taken at the reconstructed data real-space positions;
  # in this case, do: data_positions_rec = recon.read_shifted_positions('data')
  data_positions_rec = recon.read_shifted_positions(data_positions)
  # RecSym = remove large scale RSD from randoms
  randoms_positions_rec = recon.read_shifted_positions(randoms_positions)
  # or RecIso
  # randoms_positions_rec = recon.read_shifted_positions(randoms_positions, field='disp')


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
