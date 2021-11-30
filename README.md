# pyrecon - Python reconstruction code

## Introduction

**pyrecon** is a package to perform reconstruction within Python, using different algorithms, so far:

  - MultiGridReconstruction, based on Martin J. White's code https://github.com/martinjameswhite/recon_code
  - IterativeFFTParticleReconstruction, based on Julian E. Bautista's code https://github.com/julianbautista/eboss_clustering/blob/master/python/recon.py
  - IterativeFFTReconstruction, iterative algorithm of Burden et al. 2015 (https://arxiv.org/abs/1504.02591) at the field-level (as opposed to IterativeFFTParticleReconstruction)
  - PlaneParallelFFTReconstruction, base algorithm of Eisenstein et al. 2007 (https://arxiv.org/pdf/astro-ph/0604362.pdf), in the plane-parallel approximation.

With Python, a typical reconstruction run is (e.g. for MultiGridReconstruction; the same works for other algorithms):
```
from pyrecon import MultiGridReconstruction

# line-of-sight "los" can be local (None, default) or an axis, 'x', 'y', 'z', or a 3-vector
# Instead of boxsize and boxcenter, one can provide a (N, 3) array of Cartesian positions: positions=
recon = MultiGridReconstruction(f=0.8, bias=2.0, los=None, nmesh=512, boxsize=1000., boxcenter=2000.)
recon.assign_data(positions_data, weights_data) # positions_data is a (N, 3) array of Cartesian positions, weights a (N,) array
# You can skip the following line if you assume uniform selection function (randoms)
recon.assign_randoms(positions_randoms, weights_randoms)
recon.set_density_contrast()
recon.run()
# If you are using IterativeFFTParticleReconstruction, displacements are to be taken at the reconstructed data real-space positions;
# in this case, do: positions_rec_data = positions_data - recon.read_shifts('data')
positions_rec_data = positions_data - recon.read_shifts(positions_data)
# RecSym = remove large scale RSD from randoms
positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms)
# or RecIso
# positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms, field='disp')
```
Also provided a script to run reconstruction as a standalone:
```
pyrecon [-h] config-fn [--data-fn [<fits, hdf5 file>]] [--randoms-fn [<fits, hdf5 file>]] [--output-data-fn [<fits, hdf5 file>]] [--output-randoms-fn [<fits, hdf5file>]]
```
An example of configuration file is provided in [config](https://github.com/cosmodesi/pyrecon/blob/main/bin/config_example.yaml).
data-fn, randoms-fn are input data and random file names to override those in configuration file.
The same holds for output files output-data-fn, output-randoms-fn.

## In progress

Check algorithm details (see notes in docstrings).

## Documentation

Documentation is hosted on Read the Docs, [pyrecon docs](https://pyrecon.readthedocs.io/).

# Requirements

Only strict requirement is:

  - numpy

Extra requirements are:

  - [pyfftw](https://github.com/pyFFTW/pyFFTW) (for faster FFTs)
  - fitsio, h5py, astropy, scipy to run **pyrecon** as a standalone

## Installation

See [pyrecon docs](https://pyrecon.readthedocs.io/en/latest/user/building.html).

## License

**pyrecon** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/cosmodesi/pyrecon/blob/main/LICENSE).

## Credits

- Martin J. White for https://github.com/martinjameswhite/recon_code
- Julian E. Bautista for https://github.com/julianbautista/eboss_clustering/blob/master/python/recon.py
- Pedro Rangel Caetano for inspiration for the script bin/recon
- Sesh Nadathur for careful checks against Revolver https://github.com/seshnadathur/Revolver/blob/main/python_tools/recon.py
- Enrique Paillas for bug reports
- Grant Merz for propagator https://github.com/grantmerz/DESI_Recon
