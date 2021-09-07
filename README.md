# pyrecon - Python reconstruction code

## Introduction

**pyrecon** is a package to perform reconstruction within Python, using different algorithms, so far:

 - MultiGridReconstruction, based on Martin White's code https://github.com/martinjameswhite/recon_code
 - IterativeFFTReconstruction, based on Julian Bautista's code https://github.com/julianbautista/eboss_clustering/blob/master/python/recon.py


A typical reconstruction run is (e.g. for MultiGridReconstruction; the same works for other algorithms):
```
from pyrecon import MultiGridReconstruction

rec = MultiGridReconstruction(f=0.8,bias=2.0,nmesh=512,boxsize=1000.,boxcenter=2000.)
rec.assign_data(positions_data,weights_data)
rec.assign_randoms(positions_randoms,weights_randoms)
rec.set_density_contrast()
rec.run()
positions_rec_data = positions_data - rec.read_shifts(positions_data)
# RecSym = remove large scale RSD from randoms
positions_rec_randoms = positions_randoms - rec.read_shifts(positions_randoms)
# Or RecIso
# positions_rec_randoms = positions_randoms - rec.read_shifts(positions_randoms,with_rsd=False)
```

## Warning

In progress! Handling bugs with MultiGridReconstruction.
Should be solved fairly rapidly.

## Documentation

Documentation is hosted on Read the Docs, [pyrecon docs](https://pyrecon.readthedocs.io/).

# Requirements

Only strict requirements are:
- numpy

For faster FFTs:
- pyfftw

## Installation

To install the code:
```
$>  python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately):
```
$>  python setup.py develop --user
```

With Mac OS, if you wish to use clang compiler (instead of gcc), you may encounter an error related to ``-fopenmp`` flag.
In this case, you can try to export:
```
$>  export CC=clang
```
Before installing pyrecon. This will set clang OpenMP flags for compilation.
Note that with Mac OS gcc can point to clang.

## Requirements

- numpy
- pyfftw (optional, for faster FFTs)

## License

**pyrecon** is free software distributed under a GPLv3 license. For details see the [LICENSE](https://github.com/adematti/pyrecon/blob/main/LICENSE).
