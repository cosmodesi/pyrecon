import numpy as np

from pyrecon import RealMesh, ComplexMesh, MeshInfo, utils


def test_info():
    info = MeshInfo(boxsize=1000.,boxcenter=0.,nmesh=64)
    assert np.all(info.boxsize == 1000)
    info.boxsize = 2000.
    assert np.all(info.boxsize == 2000)


def sample_cic(mesh, positions, weights=1.):
    index = positions.astype(int)
    dindex = positions-index
    #edges = [scipy.linspace(0,n,n+1) for n in self.Nmesh]
    ishifts = np.array(np.meshgrid(*([[0,1]]*3),indexing='ij')).reshape((3,-1)).T
    for ishift in ishifts:
        sindex = index + ishift
        sweight = np.prod((1-dindex) + ishift*(-1+2*dindex),axis=-1)*weights
        #delta += scipy.histogramdd(sindex,bins=edges,weights=sweight)[0]
        #delta[tuple(sindex.T)] += sweight
        np.add.at(mesh,tuple(sindex.T),sweight)


def read_cic(mesh, positions):
    index = positions.astype(int)
    dindex = positions-index
    shifts = np.zeros_like(positions,shape=(len(positions),))
    ishifts = np.array(np.meshgrid(*([[0,1]]*3),indexing='ij')).reshape((3,-1)).T
    for ishift in ishifts:
        sindex = tuple((index + ishift).T)
        sweight = np.prod((1-dindex) + ishift*(-1+2*dindex),axis=-1)
        shifts += mesh[sindex]*sweight
    return shifts


def test_cic():
    mesh = RealMesh(boxsize=1000.,boxcenter=0.,nmesh=64,dtype='f8',nthreads=4)
    size = 100000
    rng = np.random.RandomState(seed=42)
    positions = np.array([rng.uniform(-400.,400.,size) for i in range(3)]).T
    weights = rng.uniform(0.5,1.,size)
    mesh.assign_cic(positions,weights)
    assert np.allclose(np.sum(mesh),np.sum(weights))
    mesh_ref = np.zeros_like(mesh)
    sample_cic(mesh_ref,((positions - mesh.boxcenter)/mesh.boxsize + 0.5)*mesh.nmesh,weights)
    assert np.allclose(mesh,mesh_ref)
    positions = np.array([rng.uniform(-400.,400.,size) for i in range(3)]).T
    shifts = mesh.read_cic(positions)
    shifts_ref = read_cic(mesh,((positions - mesh.boxcenter)/mesh.boxsize + 0.5)*mesh.nmesh)
    assert np.allclose(shifts,shifts_ref)


def test_finite_difference_cic():
    mesh = RealMesh(boxsize=1000.,boxcenter=0.,nmesh=64,dtype='f8')
    rng = np.random.RandomState(seed=42)
    mesh.value = 1.
    size = 10000
    positions = np.array([rng.uniform(-400.,400.,size) for i in range(3)]).T
    shifts = mesh.read_finite_difference_cic(positions)
    assert np.allclose(shifts,0)


def test_smoothing():
    from matplotlib import pyplot as plt
    mesh = RealMesh(boxsize=1000.,boxcenter=0.,nmesh=64,dtype='f8')
    radius = 10
    mesh.value = np.random.uniform(0.,1.,mesh.shape)
    s = np.sum(mesh)
    mesh_fft = mesh.deepcopy()
    mesh_fft.smooth_gaussian(method='fft',radius=radius)
    assert np.allclose(np.sum(mesh_fft),s)
    mesh_brute = mesh.deepcopy()
    mesh_brute.smooth_gaussian(method='real',radius=radius)
    assert np.allclose(np.sum(mesh_brute),s)
    fig,lax = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True)
    sl = slice(1,4)
    lax[0].imshow(np.mean(mesh[:,:,sl],axis=-1))
    lax[1].imshow(np.mean(mesh_fft[:,:,sl],axis=-1))
    lax[2].imshow(np.mean(mesh_brute[:,:,sl],axis=-1))
    plt.show()


def test_fft():
    mesh = RealMesh(boxsize=1000.,boxcenter=0.,nmesh=4,dtype='f8')
    mesh.value = np.random.uniform(0.,1.,mesh.shape)
    for engine in ['numpy','fftw']:
        for hermitian in [True,False]:
            mesh2 = mesh.to_complex(engine=engine,hermitian=hermitian).to_real(engine=engine)
            assert np.allclose(mesh2,mesh,atol=1e-5)


def test_hermitian():
    mesh = RealMesh(boxsize=1000.,boxcenter=0.,nmesh=4,dtype='f8')
    radius = 10
    mesh.value = np.random.uniform(0.,1.,mesh.shape)
    mesh_fft = mesh.deepcopy()
    mesh_fft.smooth_gaussian(method='fft',hermitian=False,radius=radius)
    mesh_hermitian = mesh.deepcopy()
    mesh_hermitian.smooth_gaussian(method='fft',hermitian=True,radius=radius)
    assert np.allclose(mesh_hermitian,mesh_fft)


def test_misc():

    for Cls, dtype in zip([RealMesh, ComplexMesh], ['f8', 'c16']):
        mesh = Cls(value=1., boxsize=1., boxcenter=0., nmesh=(4,3,5), dtype=dtype)
        value = np.array(mesh.value)
        arrays = [1. + c for c in mesh.coords()]
        index_zero = (0,0,0)
        tmp = sum(utils.broadcast_arrays(*arrays))
        tmp[index_zero] = 1.
        for exp in [-2, -1, 1, 2]:
            ref = value*tmp**exp
            ref[index_zero] = 0.
            mesh.value = 1.
            mesh.prod_sum(arrays, exp=exp)
            mesh[index_zero] = 0.
            assert np.allclose(mesh.value, ref)

        #for axis in range(3):
        #    mesh.value = 1.
        #    ref = np.apply_along_axis(lambda x: x*arrays[axis], axis, mesh.value)
        #    mesh.prod_along_axis(arrays[axis], axis=axis)
        #    assert np.allclose(mesh.value, ref)


def test_pyfftw():
    import time
    import pyfftw

    nthreads = 4
    nmesh = (256,)*3
    niter = 10
    rho = pyfftw.empty_aligned(nmesh, dtype='complex128')
    rhok = pyfftw.empty_aligned(nmesh, dtype='complex128')

    fft_obj = pyfftw.FFTW(rho, rhok, axes=[0, 1, 2], threads=nthreads)
    ifft_obj = pyfftw.FFTW(rhok, rho, axes=[0, 1, 2], threads=nthreads, direction='FFTW_BACKWARD')

    t0 = time.time()
    for i in range(niter): fft_obj(input_array=rho, output_array=rhok)
    print('Took {:.3f} s.'.format(time.time() - t0))
    #ifft_obj(input_array=rhok, output_array=rho)

    rho2 = rho.copy()
    rhok2 = rhok.copy()
    t0 = time.time()
    for i in range(niter): fft_obj(input_array=rho2, output_array=rhok2)
    print('Took {:.3f} s.'.format(time.time() - t0))


def test_timing():

    import os
    import time
    nmesh = 400
    nthreads = 2
    os.environ['OMP_NUM_THREADS'] = str(nthreads)
    niter = 10

    mesh = ComplexMesh(1., boxsize=1000., boxcenter=0., nmesh=nmesh, hermitian=False, dtype='c16', nthreads=nthreads)
    t0 = time.time()
    for i in range(niter):
        k = utils.broadcast_arrays(*mesh.coords())
        k2 = sum(kk**2 for kk in k)
        k2[0,0,0] = 1. # to avoid dividing by 0
        mesh /= k2
    print('numpy took {:.3f} s.'.format(time.time() - t0))

    mesh = ComplexMesh(1., boxsize=1000., boxcenter=0., nmesh=nmesh, hermitian=False, dtype='c16', nthreads=nthreads)
    t0 = time.time()
    for i in range(niter):
        k = mesh.coords()
        k2 = [k**2 for k in mesh.coords()]
        mesh.prod_sum(k2, exp=-1)
        mesh[0,0,0] = 0.
    print('C took {:.3f} s.'.format(time.time() - t0))

    import sys
    sys.path.insert(0,'../../../../reconstruction/Revolver')
    from python_tools import fastmodules
    mesh = ComplexMesh(1., boxsize=1000., boxcenter=0., nmesh=nmesh, hermitian=False, dtype='c16', nthreads=nthreads)
    t0 = time.time()
    for i in range(niter):
        k = mesh.coords()[0]
        fastmodules.divide_k2(mesh.value, mesh.value, k)
        mesh[0,0,0] = 0.
    print('fastmodules took {:.3f} s.'.format(time.time() - t0))

    bias = 2.
    mesh = ComplexMesh(1., boxsize=1000., boxcenter=0., nmesh=nmesh, hermitian=False, dtype='c16', nthreads=nthreads)
    t0 = time.time()
    for i in range(niter):
        k = utils.broadcast_arrays(*mesh.coords())[0]
        k[0,0,0] = 1.
        mesh.value *= 1j/(bias*k)
    print('numpy took {:.3f} s.'.format(time.time() - t0))

    bias = 2.
    mesh = ComplexMesh(1., boxsize=1000., boxcenter=0., nmesh=nmesh, hermitian=False, dtype='c16', nthreads=nthreads)
    t0 = time.time()
    for i in range(niter):
        k = utils.broadcast_arrays(*mesh.coords())[0]
        k[0,0,0] = 1.
        mesh.value = mesh.value[...]
    print('numpy took {:.3f} s.'.format(time.time() - t0))

    mesh = ComplexMesh(1., boxsize=1000., boxcenter=0., nmesh=nmesh, hermitian=False, dtype='c16', nthreads=nthreads)
    t0 = time.time()
    for i in range(niter):
        k = mesh.coords()[0]
        k[0] = 1.
        fastmodules.mult_kx(mesh.value, mesh.value, k, bias)
    print('fastmodules took {:.3f} s.'.format(time.time() - t0))




if __name__ == '__main__':

    #test_pyfftw()
    #test_timing()
    #test_misc()
    #exit()
    test_info()
    test_cic()
    test_finite_difference_cic()
    test_smoothing()
    test_fft()
    test_hermitian()
    test_misc()
