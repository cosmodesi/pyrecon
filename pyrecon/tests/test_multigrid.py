import os
import time
import subprocess

import numpy as np
import fitsio

from pyrecon.multigrid import OriginalMultiGridReconstruction, MultiGridReconstruction
from pyrecon.utils import distance, cartesian_to_sky, DistanceToRedshift


def get_random_catalog(size=100000, boxsize=1000., seed=None):
    rng = np.random.RandomState(seed=seed)
    positions = np.array([rng.uniform(0.,1.,size) for i in range(3)]).T*boxsize
    weights = rng.uniform(0.5,1.,size)
    return {'Position':positions,'Weight':weights}


def test_random():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=84)
    recon = MultiGridReconstruction(f=0.8,bias=2.,nthreads=1,positions=randoms['Position'],nmesh=8,dtype='f8')
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])
    recon.set_density_contrast()
    recon.run(jacobi_niterations=1,vcycle_niterations=1)
    #recon.run()
    #print(recon.read_shifts(data['Position']))
    assert np.all(np.abs(recon.read_shifts(data['Position'])) < 2.)


def test_no_nrandoms():
    boxsize = 1000.
    data = get_random_catalog(boxsize=boxsize,seed=42)
    recon = MultiGridReconstruction(f=0.8,bias=2.,los='x',nthreads=4,boxcenter=boxsize/2.,boxsize=boxsize,nmesh=8,dtype='f8')
    recon.assign_data(data['Position'],data['Weight'])
    assert not recon.has_randoms
    recon.set_density_contrast()
    assert np.allclose(np.mean(recon.mesh_delta), 0.)
    recon.run(jacobi_niterations=1,vcycle_niterations=1)
    #recon.run()
    assert np.all(np.abs(recon.read_shifts(data['Position'])) < 2.)


def test_dtype():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=84)
    recon_f4 = MultiGridReconstruction(f=0.8,bias=2.,nthreads=4,positions=randoms['Position'],nmesh=64,dtype='f4')
    recon_f4.assign_data(data['Position'],data['Weight'])
    recon_f4.assign_randoms(randoms['Position'],randoms['Weight'])
    recon_f4.set_density_contrast()
    recon_f4.run()
    recon_f8 = MultiGridReconstruction(f=0.8,bias=2.,nthreads=4,positions=randoms['Position'],nmesh=64,dtype='f8')
    recon_f8.assign_data(data['Position'],data['Weight'])
    recon_f8.assign_randoms(randoms['Position'],randoms['Weight'])
    recon_f8.set_density_contrast()
    recon_f8.run()
    assert not np.all(recon_f4.mesh_phi == recon_f8.mesh_phi)
    assert np.allclose(recon_f4.mesh_phi,recon_f8.mesh_phi,rtol=1e-2,atol=1e-2)


def test_dtype():
    data = get_random_catalog(seed=42)
    randoms = get_random_catalog(seed=84)
    recon_f4 = MultiGridReconstruction(f=0.8,bias=2.,nthreads=4,positions=randoms['Position'],nmesh=64,dtype='f4')
    recon_f4.assign_data(data['Position'],data['Weight'])
    recon_f4.assign_randoms(randoms['Position'],randoms['Weight'])
    recon_f4.set_density_contrast()
    recon_f4.run()
    shifts_f4 = recon_f4.read_shifts(data['Position'],with_rsd=True)
    recon_f8 = MultiGridReconstruction(f=0.8,bias=2.,nthreads=4,positions=randoms['Position'],nmesh=64,dtype='f8')
    recon_f8.assign_data(data['Position'],data['Weight'])
    recon_f8.assign_randoms(randoms['Position'],randoms['Weight'])
    recon_f8.set_density_contrast()
    recon_f8.run()
    shifts_f8 = recon_f8.read_shifts(data['Position'],with_rsd=True)
    assert not np.all(shifts_f4 == shifts_f8)
    assert np.allclose(shifts_f4,shifts_f8,rtol=1e-2,atol=1e-2)


def test_los():
    boxsize = 1000.
    boxcenter = [boxsize/2]*3
    data = get_random_catalog(boxsize=boxsize,seed=42)
    randoms = get_random_catalog(boxsize=boxsize,seed=84)
    recon = MultiGridReconstruction(f=0.8,bias=2.,los='x',nthreads=4,boxcenter=boxcenter,boxsize=boxsize,nmesh=64,dtype='f8')
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])
    recon.set_density_contrast()
    recon.run()
    shifts_global = recon.read_shifts(data['Position'],with_rsd=True)
    offset = 1e8
    boxcenter[0] += offset
    data['Position'][:,0] += offset
    randoms['Position'][:,0] += offset
    recon = MultiGridReconstruction(f=0.8,bias=2.,nthreads=4,boxcenter=boxcenter,boxsize=boxsize,nmesh=64,dtype='f8')
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])
    recon.set_density_contrast()
    recon.run()
    shifts_local = recon.read_shifts(data['Position'],with_rsd=True)
    assert np.allclose(shifts_local,shifts_global,rtol=1e-3,atol=1e-3)


def compute_ref(data_fn, randoms_fn, output_data_fn, output_randoms_fn):

    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70,Om0=0.3)

    def comoving_distance(z):
        return cosmo.comoving_distance(z).value*cosmo.h

    input_fn = [fn.replace('.fits','.rdzw') for fn in [data_fn,randoms_fn]]

    for fn,infn in zip([data_fn,randoms_fn],input_fn):
        catalog = fitsio.read(fn)
        #distance, ra, dec = cartesian_to_sky(catalog['Position'])
        #rdzw = [ra,dec,DistanceToRedshift(comoving_distance)(distance)] + [catalog['Weight']]
        rdzw = list(catalog['Position'].T) + [catalog['Weight']]
        np.savetxt(infn,np.array(rdzw).T)

    catalog_dir = os.path.dirname(infn)
    command = '{0} {1} {2} {2} 2 0.81 15'.format(recon_code,*[os.path.basename(infn) for infn in input_fn])
    subprocess.call(command,shell=True,cwd=catalog_dir)

    output_fn = [os.path.join(catalog_dir,base) for base in ['data_rec.xyzw','rand_rec.xyzw']]
    for infn,fn,outfn in zip([data_fn,randoms_fn],[output_data_fn,output_randoms_fn],output_fn):
        x,y,z,w = np.loadtxt(outfn,unpack=True)
        positions = np.array([x,y,z]).T
        catalog = fitsio.read(infn)
        #print(np.mean(distance(positions-catalog['Position'])))
        catalog['Position'] = positions
        catalog['Weight'] = w
        fitsio.write(fn,catalog,clobber=True)


def test_recon(data_fn, randoms_fn, output_data_fn, output_randoms_fn):
    #boxsize = 1199.9995117188 in float32
    #boxcenter = [1753.8884277344,400.0001831055,400.0003662109] in float64
    boxsize = 1199.9993880913
    boxcenter = [1753.8883893991,400.0001954356,400.0003824141]
    recon = OriginalMultiGridReconstruction(nthreads=2,boxsize=boxsize,boxcenter=boxcenter,nmesh=128,dtype='f8')
    recon.set_cosmo(f=0.81,bias=2.)
    """
    recon = OriginalMultiGridReconstruction(nthreads=1,positions=fitsio.read(randoms_fn,columns=['Position'])['Position'],nmesh=128,dtype='f4')
    recon.set_cosmo(f=0.81,bias=2.)
    print(recon.mesh_data.boxsize,recon.mesh_data.boxcenter)
    """
    ext = 1
    nslabs = 1
    for fn,assign in zip([data_fn,randoms_fn],[recon.assign_data,recon.assign_randoms]):
        with fitsio.FITS(fn,'r') as ff:
            ff = ff[ext]
            size = ff.get_nrows()
            for islab in range(nslabs):
                start = islab*size//nslabs
                stop = (islab+1)*size//nslabs
                data = ff.read(columns=['Position','Weight'],rows=range(start,stop))
                assign(data['Position'],data['Weight'])


    recon.set_density_contrast()
    #print(np.max(recon.mesh_delta))
    recon.run()
    #print(np.std(recon.mesh_phi))
    #recon.f = recon.beta

    for input_fn,output_fn in zip([data_fn,randoms_fn],[output_data_fn,output_randoms_fn]):
        with fitsio.FITS(input_fn,'r') as ffin:
            ffin = ffin[ext]
            size = ffin.get_nrows()
            with fitsio.FITS(output_fn,'rw',clobber=True) as ffout:
                for islab in range(nslabs):
                    start = islab*size//nslabs
                    stop = (islab+1)*size//nslabs
                    data = ffin.read(rows=range(start,stop))
                    shifts = recon.read_shifts(data['Position'],with_rsd=True)
                    print('RMS',(np.mean(np.sum(shifts**2,axis=-1))/3)**0.5)
                    data['Position'] -= shifts
                    if islab == 0: ffout.write(data)
                    else: ffout[-1].append(data)


def compare_ref(data_fn, output_data_fn, ref_output_data_fn):
    positions = fitsio.read(data_fn)['Position']
    output_positions = fitsio.read(output_data_fn)['Position']
    ref_output_positions = fitsio.read(ref_output_data_fn)['Position']

    print('abs test - ref',np.max(distance(output_positions-ref_output_positions)))
    print('rel test - ref',np.max(distance(output_positions-ref_output_positions)/distance(ref_output_positions-positions)))
    print('test',np.mean(distance(output_positions-positions)))
    print('ref',np.mean(distance(ref_output_positions-positions)))
    assert np.allclose(output_positions,ref_output_positions,rtol=1e-7,atol=1e-7)


def test_script(data_fn, randoms_fn, output_data_fn, output_randoms_fn):

    catalog_dir = '_catalogs'
    command = 'pyrecon config_multigrid.yaml --data-fn {} --randoms-fn {} --output-data-fn {} --output-randoms-fn {}'.format(
                os.path.relpath(data_fn,catalog_dir),os.path.relpath(randoms_fn,catalog_dir),
                os.path.relpath(output_data_fn,catalog_dir),os.path.relpath(output_randoms_fn,catalog_dir))
    subprocess.call(command,shell=True)
    data = fitsio.read(data_fn,columns=['Position','Weight'])
    randoms = fitsio.read(randoms_fn,columns=['Position','Weight'])
    recon = MultiGridReconstruction(nthreads=4,positions=randoms['Position'],nmesh=128,dtype='f8')
    recon.set_cosmo(f=0.8,bias=2.)
    recon.assign_data(data['Position'],data['Weight'])
    recon.assign_randoms(randoms['Position'],randoms['Weight'])

    recon.set_density_contrast()
    recon.run()

    ref_positions_rec_data = data['Position'] - recon.read_shifts(data['Position'])
    ref_positions_rec_randoms = randoms['Position'] - recon.read_shifts(randoms['Position'])

    data = fitsio.read(output_data_fn,columns=['Position_rec'])
    randoms = fitsio.read(output_randoms_fn,columns=['Position_rec'])

    #print(ref_positions_rec_data,data['Position_rec'],ref_positions_rec_data-data['Position_rec'])
    assert np.allclose(ref_positions_rec_data,data['Position_rec'])
    assert np.allclose(ref_positions_rec_randoms,randoms['Position_rec'])


def test_script_no_randoms(data_fn, output_data_fn):

    catalog_dir = '_catalogs'
    command = 'pyrecon config_multigrid_no_randoms.yaml --data-fn {} --output-data-fn {}'.format(
                os.path.relpath(data_fn,catalog_dir),os.path.relpath(output_data_fn,catalog_dir))
    subprocess.call(command,shell=True)
    data = fitsio.read(data_fn)
    boxsize = 800
    boxcenter = boxsize/2.
    recon = MultiGridReconstruction(nthreads=4,los='x',boxcenter=boxcenter,boxsize=boxsize,nmesh=128,dtype='f8')
    recon.set_cosmo(f=0.8,bias=2.)
    recon.assign_data(data['RSDPosition'])
    recon.set_density_contrast()
    recon.run()

    ref_positions_rec_data = data['RSDPosition'] - recon.read_shifts(data['RSDPosition'])
    #velocityoffset = recon.read_shifts(data['RSDPosition'],with_rsd=True) - recon.read_shifts(data['RSDPosition'],with_rsd=False)
    #print(velocityoffset)
    #print(data['VelocityOffset'])
    #assert np.all(np.abs(velocityoffset[:,0] - data['VelocityOffset'][:,0]) < np.abs(data['VelocityOffset'][:,0]))

    data = fitsio.read(output_data_fn,columns=['Position_rec'])

    #print(ref_positions_rec_data,data['Position_rec'],ref_positions_rec_data-data['Position_rec'])
    assert np.allclose(ref_positions_rec_data,data['Position_rec'])


def compute_power(*list_data_randoms):

    from matplotlib import pyplot as plt
    from nbodykit.lab import FITSCatalog, FKPCatalog, ConvolvedFFTPower

    for linestyle,(data_fn,randoms_fn) in zip(['-','--'],list_data_randoms):

        data = FITSCatalog(data_fn)
        randoms = FITSCatalog(randoms_fn)

        for catalog in [data,randoms]:
            catalog['WEIGHT_FKP'] = np.ones(catalog.size,dtype='f8')
            catalog['WEIGHT_COMP'] = catalog['Weight']

        fkp = FKPCatalog(data,randoms)
        BoxSize = 3000.
        Nmesh = 128
        ells = (0,2,4)
        mesh = fkp.to_mesh(position='Position',fkp_weight='WEIGHT_FKP',comp_weight='WEIGHT_COMP',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True,compensated=True)
        power = ConvolvedFFTPower(mesh,poles=ells,kmin=0.,dk=0.01)
        poles = power.poles

        for ill,ell in enumerate(ells):
            pk = poles['power_{:d}'.format(ell)] - power.attrs['shotnoise'] if ell == 0 else poles['power_{:d}'.format(ell)]
            plt.plot(poles['k'],poles['k']*pk,color='C{:d}'.format(ill),linestyle=linestyle)

    plt.xlabel('$k$ [$h/\mathrm{Mpc}$]')
    plt.ylabel('$kP(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    plt.show()


def compute_power_no_randoms(list_data, list_positions):

    from matplotlib import pyplot as plt
    from nbodykit.lab import FITSCatalog, FFTPower

    for linestyle,data_fn,position in zip(['-','--'],list_data,list_positions):

        data = FITSCatalog(data_fn)
        BoxSize = 800.
        Nmesh = 128
        ells = (0,2,4)
        mesh = data.to_mesh(position=position,BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True,compensated=True)
        power = FFTPower(mesh,mode='2d',poles=ells,kmin=0.,dk=0.01,los=[1,0,0])
        poles = power.poles

        for ill,ell in enumerate(ells):
            pk = poles['power_{:d}'.format(ell)] - power.attrs['shotnoise'] if ell == 0 else poles['power_{:d}'.format(ell)]
            plt.plot(poles['k'],poles['k']*pk,color='C{:d}'.format(ill),linestyle=linestyle)

    plt.xlabel('$k$ [$h/\mathrm{Mpc}$]')
    plt.ylabel('$kP(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    plt.show()


if __name__ == '__main__':

    #with MemoryMonitor() as mem:
    #    for i in range(2):
    #        test_random()
    import utils
    from utils import box_data_fn, data_fn, randoms_fn, catalog_dir, MemoryMonitor
    from pyrecon.utils import setup_logging

    setup_logging()
    # Uncomment to compute catalogs needed for these tests
    utils.setup()

    recon_code = os.path.join(os.path.abspath(os.path.dirname(__file__)),'_codes','recon')
    output_data_fn = os.path.join(catalog_dir,'data_rec.fits')
    output_randoms_fn = os.path.join(catalog_dir,'randoms_rec.fits')
    ref_output_data_fn = os.path.join(catalog_dir,'ref_data_rec.fits')
    ref_output_randoms_fn = os.path.join(catalog_dir,'ref_randoms_rec.fits')
    script_output_box_data_fn = os.path.join(catalog_dir,'script_box_data_rec.fits')
    script_output_data_fn = os.path.join(catalog_dir,'script_data_rec.fits')
    script_output_randoms_fn = os.path.join(catalog_dir,'script_randoms_rec.fits')
    
    test_random()
    test_no_nrandoms()
    test_dtype()
    test_los()
    test_recon(data_fn,randoms_fn,output_data_fn,output_randoms_fn)
    compute_ref(data_fn,randoms_fn,ref_output_data_fn,ref_output_randoms_fn)
    compare_ref(data_fn,output_data_fn,ref_output_data_fn)
    compare_ref(randoms_fn,output_randoms_fn,ref_output_randoms_fn)
    test_script(data_fn,randoms_fn,script_output_data_fn,script_output_randoms_fn)
    test_script_no_randoms(box_data_fn, script_output_box_data_fn)
    #compute_power_no_randoms([script_output_box_data_fn]*2, ['RSDPosition','Position_rec'])
    #compute_power((data_fn,randoms_fn),(output_data_fn,output_randoms_fn))
    #compute_power((data_fn,randoms_fn),(ref_output_data_fn,ref_output_randoms_fn))
    #compute_power((ref_output_data_fn,ref_output_randoms_fn),(output_data_fn,output_randoms_fn))
