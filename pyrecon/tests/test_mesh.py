import numpy as np

from pyrecon import RealMesh, MeshInfo, utils


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


def test_smoothing():
    from matplotlib import pyplot as plt
    mesh = RealMesh(boxsize=1000.,boxcenter=0.,nmesh=64,dtype='f8')
    radius = 10
    mesh.value = np.random.uniform(0.,1.,mesh.shape)
    s = np.sum(mesh)
    mesh_fft = mesh.deepcopy()
    mesh_fft.smooth_gaussian(method='fft',radius=radius)
    print(np.sum(mesh_fft)/s)
    mesh_brute = mesh.deepcopy()
    mesh_brute.smooth_gaussian(method='bruteforce',radius=radius)
    print(np.sum(mesh_brute)/s)
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





if __name__ == '__main__':

    #test_info()
    #test_cic()
    #test_smoothing()
    #test_fft()
    test_hermitian()
