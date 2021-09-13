.. title:: pyrecon docs

**************************************
Welcome to pyrecon's documentation!
**************************************

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

**pyrecon** is a Python package to perform reconstruction in BAO analyses with various algorithms.

A typical reconstruction run is (e.g. for MultiGridReconstruction; the same works for other algorithms):

.. code-block:: python

  from pyrecon import MultiGridReconstruction

  recon = MultiGridReconstruction(f=0.8,bias=2.0,nmesh=512,boxsize=1000.,boxcenter=2000.)
  recon.assign_data(positions_data,weights_data)
  recon.assign_randoms(positions_randoms,weights_randoms)
  recon.set_density_contrast()
  recon.run()
  positions_rec_data = positions_data - recon.read_shifts(positions_data)
  # RecSym = remove large scale RSD from randoms
  positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms)
  # Or RecIso
  # positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms,with_rsd=False)


**************
Code structure
**************

The code structure is the following:

  - recon.py implements the base reconstruction class
  - mesh.py implements mesh object, with FFT engines
  - utils.py implements various utilities
  - a module for each algorithm


Changelog
=========

* :doc:`developer/changes`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
