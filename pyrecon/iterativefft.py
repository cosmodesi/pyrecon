import numpy as np

from .recon import BaseReconstruction
from . import utils


class IterativeFFTReconstruction(BaseReconstruction):
    """
    Implementation of Bautista 2018 algorithm.
    So far we sick to the implementation at https://github.com/julianbautista/eboss_clustering/blob/master/python/recon.py
    """
    def __init__(self, fft_engine='numpy', fft_wisdom=None, **kwargs):
        """
        Initialize :class:`IterativeFFTReconstruction`.

        Parameters
        ----------
        fft_engine : string, BaseFFTEngine, default='numpy'
            Engine for fast Fourier transforms. See :class:`BaseFFTEngine`.

        fft_wisdom : string, tuple
            Wisdom for FFTW, if ``fft_engine`` is 'fftw'.

        kwargs : dict
            See :class:`BaseReconstruction` for parameters.
        """
        super(IterativeFFTReconstruction,self).__init__(**kwargs)
        kwargs = {}
        if fft_wisdom is not None: kwargs['wisdom'] = 'fft_wisdom'
        kwargs['hermitian'] = False
        self.fft_engine = self.mesh_data.get_fft_engine(fft_engine,**kwargs)

    def assign_data(self, positions, weights=None):
        """
        Assign (paint) data to :attr:`mesh_data`.
        Keeps track of input positions (for :meth:`run`) and weights (for :meth:`set_density_contrast`).
        See :meth:`BaseReconstruction.assign_data` for parameters.
        """
        if weights is None:
            weights = np.ones_like(positions,shape=(len(positions),))
        if self.mesh_data.value is None:
            self._positions_data = positions
            self._weights_data = weights
        else:
            self._positions_data = np.concatenate([self._positions_data,positions],axis=0)
            self._weights_data = np.concatenate([self._weights_data,weights],axis=0)
        self.mesh_data.assign_cic(positions,weights=weights)

    def assign_randoms(self, positions, weights=None):
        """
        Assign (paint) data to :attr:`mesh_data`.
        Keeps track of sum of weights (for :meth:`set_density_contrast`).
        See :meth:`BaseReconstruction.assign_randoms` for parameters.
        """
        if weights is None:
            weights = np.ones_like(positions,shape=(len(positions),))
        if self.mesh_randoms.value is None:
            self._sum_randoms = 0.
            self._size_randoms = 0
        #super(IterativeFFTReconstruction,self).assign_randoms(positions,weights=weights)
        self.mesh_randoms.assign_cic(positions,weights=weights)
        self._sum_randoms += np.sum(weights)
        self._size_randoms += len(positions)

    def set_density_contrast(self, ran_min=0.01, smoothing_radius=15.):
        """
        Set :math:`mesh_delta` field :attr:`mesh_delta` from data and randoms fields :attr:`mesh_data` and :attr:`mesh_randoms`.

        Note
        ----
        This method follows Julian's reconstruction code.
        Handling of ``ran_min`` is better than in :meth:`BaseReconstruction.set_density_contrast`.
        :attr:`mesh_data` and :attr:`mesh_randoms` fields are assumed to be smoothed already.

        Parameters
        ----------
        ran_min : float, default=0.01
            :attr:`mesh_randoms` points below this threshold times mean random weights have their density contrast set to 0.
        """
        self.ran_min = ran_min
        self.smoothing_radius = smoothing_radius
        alpha = np.sum(self._weights_data)*1./self._sum_randoms
        self.mesh_delta = self.mesh_data - alpha*self.mesh_randoms
        mask = self.mesh_randoms > ran_min * self._sum_randoms/self._size_randoms
        self.mesh_delta[mask] /= (self.bias*alpha*self.mesh_randoms[mask])
        self.mesh_delta[~mask] = 0.

    def run(self, niterations=3, **kwargs):
        """
        Run reconstruction, i.e. compute reconstructed data real-space positions (:attr:`_positions_rec_data`)
        and Zeldovich displacements fields :attr:`psi`.

        Parameters
        ----------
        niterations : int
            Number of iterations.
        """
        self._iter = 0
        # Gaussian smoothing before density contrast calculation
        self.mesh_data.smooth_gaussian(self.smoothing_radius,method='fft',engine=self.fft_engine)
        self.mesh_randoms.smooth_gaussian(self.smoothing_radius,method='fft',engine=self.fft_engine)
        self._positions_rec_data = self._positions_data.copy()
        for iter in range(niterations):
            self.psi = self._iterate(return_psi=iter==niterations-1)

    def _iterate(self, return_psi=False):
        self.log_info('Running iteration {:d}.'.format(self._iter))

        if self._iter > 0:
            self.mesh_data = self.mesh_delta.copy(value=None)
            # Painting reconstructed data real-space positions
            super(IterativeFFTReconstruction,self).assign_data(self._positions_rec_data,weights=self._weights_data) # super in order not to save positions_rec_data
            # Gaussian smoothing before density contrast calculation
            self.mesh_data.smooth_gaussian(self.smoothing_radius,method='fft',engine=self.fft_engine)

        self.set_density_contrast(ran_min=self.ran_min)
        del self.mesh_data
        deltak = self.mesh_delta.to_complex(engine=self.fft_engine)
        k = deltak.freq()
        norm2 = sum(k_**2 for k_ in utils.broadcast_arrays(*k))
        norm2[0,0,0] = 1.
        deltak /= norm2
        deltak[0,0,0] = 0.
        self.log_info('Computing displacement field.')
        shifts = np.empty_like(self._positions_rec_data)
        psis = []
        ndim = len(k)
        for iaxis in range(ndim):
            sl = [None]*ndim; sl[iaxis] = slice(None)
            tmp = deltak*1j*k[iaxis][tuple(sl)]

            psi = tmp.to_real(engine=self.fft_engine)
            # Reading shifts at reconstructed data real-space positions
            shifts[:,iaxis] = psi.read_cic(self._positions_rec_data)
            if return_psi: psis.append(psi)

        #self.log_info('A few displacements values:')
        #for s in shifts[:3]: self.log_info('{}'.format(s))
        los = self._positions_data/utils.distance(self._positions_data)[:,None]
        # Comments in Julian's code:
        # For first loop need to approximately remove RSD component from psi to speed up calculation
        # See Burden 2015: 1504.02591v2, eq. 12 (flat sky approximation)
        if self._iter == 0:
            shifts -= self.beta/(1+self.beta)*np.sum(shifts*los,axis=-1)[:,None]*los
        # Comments in Julian's code:
        # Remove RSD from original positions of galaxies to give new positions
        # these positions are then used in next determination of psi,
        # assumed to not have RSD.
        # The iterative procedure then uses the new positions as if they'd been read in from the start
        self._positions_rec_data = self._positions_data - self.f*np.sum(shifts*los,axis=-1)[:,None]*los
        self._iter += 1
        if return_psi:
            return psis

    def read_shifts(self, positions, with_rsd=True):
        """
        Read Zeldovich displacement at input positions.

        Note
        ----
        Data shifts are read at the reconstructed real-space positions,
        while random shifts are read at the redshift-space positions, is that consistent?

        Parameters
        ----------
        positions : array of shape (N,3), string
            Cartesian positions.
            Pass string 'data' if you wish to get the displacements for the input data positions.
            Note that in this case, shifts are read at the reconstructed data real-space positions.

        with_rsd : bool, default=True
            Whether (``True``) or not (``False``) to include RSD in the shifts.
        """
        def read_cic(positions):
            shifts = np.empty_like(positions)
            for iaxis,psi in enumerate(self.psi):
                shifts[:,iaxis] = psi.read_cic(positions)
            return shifts

        if isinstance(positions,str) and positions == 'data':
            if with_rsd:
                return self._positions_data - self._positions_rec_data + read_cic(self._positions_rec_data)
            return read_cic(self._positions_rec_data)

        los = positions/utils.distance(positions)[:,None]
        shifts = read_cic(positions)
        if with_rsd:
            los = positions/utils.distance(positions)[:,None]
            shifts += self.f*np.sum(shifts*los,axis=-1)[:,None]*los
        return shifts
