import re

import numpy as np

from pyrecon import utils
from pyrecon.utils import DistanceToRedshift


def decode_eval_str(s):
    # change ${col} => col, and return list of columns
    toret = str(s)
    columns = []
    for replace in re.finditer('(\${.*?})',s):
        value = replace.group(1)
        col = value[2:-1]
        toret = toret.replace(value,col)
        if col not in columns: columns.append(col)
    return toret, columns


def test_decode_eval_str():
    s = '(${RA}>0.) & (${RA}<30.) & (${DEC}>0.) & (${DEC}<30.)'
    s,cols = decode_eval_str(s)
    print(s,cols)


def test_distance_to_redshift():
    def distance(z): return z**2
    d2z = DistanceToRedshift(distance)
    z = np.linspace(0., 20., 200)
    d = distance(z)
    assert np.allclose(d2z(d), z)
    for itemsize in [4, 8]:
        assert d2z(d.astype('f{:d}'.format(itemsize))).itemsize == itemsize


def test_random():
    positions = utils.random_box_positions(10., boxcenter=5., size=100, dtype='f4')
    assert positions.shape[0] == 100
    assert positions.dtype.itemsize == 4
    assert (positions.min() >= 0.) and (positions.max() <= 10.)
    positions = utils.random_box_positions(10., nbar=2)
    assert positions.shape[0] == 2000
    assert (positions.min() >= -5.) and (positions.max() <= 5.)


if __name__ == '__main__':

    test_decode_eval_str()
    test_distance_to_redshift()
    test_random()
