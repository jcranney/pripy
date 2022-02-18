try:
    import cupy as cp
except ImportError as ie:
    Warning("No cupy available, using basic numpy")
    import numpy as cp

import numpy as np

from scipy.special import factorial
import scipy.linalg as la

class TaylorHModel:
    def __init__(self,order,nmeas,nstate):
        self._order  = order
        self._nmeas  = nmeas
        self._nstate = nstate
        self._einbcd = "abcdefghijklmnopqrstuvwxyz"
        self.dny     = []
        self.g       = None
        assert order >= 0        
        if order > 5:
            Warning("High order taylor estimate, are you sure?")

        for ni in range(order+1):
            self.dny.append(cp.zeros([nmeas]+[nstate]*ni))
        self._einstring_eval = ["a"+self._einbcd[1:ni+1]+"".join([","+x for x in self._einbcd[1:ni+1]])+"->a" for ni in range(self._order+1)]
        self._einstring_jac  = ["ab"+self._einbcd[2:ni+2]+"".join([","+x for x in self._einbcd[2:ni+2]])+"->ab" for ni in range(self._order+1)]
        self._einstring_hess = ["abc"+self._einbcd[3:ni+3]+"".join([","+x for x in self._einbcd[3:ni+3]])+"->abc" for ni in range(self._order+1)]
        self.tot_flux = 1.0

    def set_dny(self,dny,n):
        assert dny.shape==self.dny[n].shape
        if type(dny) is cp.ndarray:
            self.dny[n] = dny.copy()
        else:
            self.dny[n][:] = cp.array(dny)
    
    def set_tot_flux(self,flux):
        self.tot_flux = flux

    def eval(self,x,order=None):
        if order is None:
            order = self._order
        if type(x) is np.ndarray:
            CPU = True
            x = cp.array(x)
        elif type(x) is cp.ndarray:
            CPU = False
        else:
            raise TypeError("input x must be either numpy or cupy array")
        out = self.dny[0].copy()
        for ni in range(1,order+1):
            out += (1/factorial(ni))*cp.einsum(self._einstring_eval[ni],self.dny[ni],*([x]*ni))
        if CPU:
            return out.get()
        else:
            return out
    
    def jacobian(self,x,order=None): 
        if order is None:
            order = self._order
        if type(x) is np.ndarray:
            CPU = True
            x = cp.array(x)
        elif type(x) is cp.ndarray:
            CPU = False
        else:
            raise TypeError("input x must be either numpy or cupy array")
        out = self.dny[1].copy()
        for ni in range(1,order+1-1):
            out += (1/factorial(ni))*cp.einsum(self._einstring_jac[ni],self.dny[ni+1],*([x]*ni))
        if CPU:
            return out.get()
        else:
            return out
        
    def hessian(self,x,order=None): 
        if order is None:
            order = self._order
        if type(x) is np.ndarray:
            CPU = True
            x = cp.array(x)
        elif type(x) is cp.ndarray:
            CPU = False
        else:
            raise TypeError("input x must be either numpy or cupy array")
        out = self.dny[2].copy()
        for ni in range(1,order+1-2):
            out += (1/factorial(ni))*cp.einsum(self._einstring_hess[ni],self.dny[ni+2],*([x]*ni))
        if CPU:
            return out.get()
        else:
            return out

    def ceo_build_dnys(self,gmt,src,imgr):
        n_px_fft = imgr.DFT_osf*imgr.N_PX_PUPIL
        pup = src.amplitude.host()
        xx,yy = np.meshgrid(np.arange(pup.shape[0]),np.arange(pup.shape[0]))

        pup_mask = pup==1

        phi = pup.copy()

        dft_matrix = la.dft(n_px_fft)
        dft_matrix = np.concatenate([dft_matrix[n_px_fft//2:,:],dft_matrix[:n_px_fft//2,]],axis=0)
        dft_matrix = dft_matrix[:,:src.phase.shape[0]]
        dft_matrix = dft_matrix[(n_px_fft-imgr.N_PX_IMAGE)//2+1:(n_px_fft+imgr.N_PX_IMAGE)//2+1,:]
        dft_matrix = np.kron(dft_matrix,dft_matrix)
        dft_matrix = dft_matrix[:,pup_mask.flatten()]

        ~gmt

        mode_to_phase = np.zeros([pup_mask.sum(),self._nstate])

        state = gmt.state
        x = state["M1"]["Txyz"][:,2]
        x *= 0.0
        gmt^=state
        ~imgr
        +src
        
        for i in range(self._nstate):
            x *= 0.0
            delta = 1e-7
            x[i] = delta
            gmt^=state
            +src
            phi = src.phase.host()
            mode_to_phase[:,i] = phi[pup_mask]/delta

        wavelength = src.wavelength
        self.wavelength = wavelength

        g0 = np.exp((1j*2*np.pi*(xx+yy)/pup.shape[0]/4))[pup_mask]
        self.g = lambda x: np.exp((1j*2*np.pi)*(mode_to_phase@x/wavelength+g0))
        self.h_true = lambda x: np.abs(dft_matrix @ self.g(x))**2
        self._mode_to_phase = mode_to_phase        
        self._dft_matrix    = dft_matrix

        d0h = np.abs(dft_matrix @ g0)**2
        self.set_dny(d0h,0)
        if self._order < 1:
            return
        
        d1h = np.zeros([imgr.N_PX_IMAGE**2,self._nstate])
        for ell1 in range(self._nstate):
            m_ell_ = mode_to_phase[:,ell1]
            d1h[:,ell1] = 2*(2*np.pi/wavelength)*((-1j)*((dft_matrix@(g0*m_ell_))).conj()*(dft_matrix@g0)).real
        self.set_dny(d1h,1)
        if self._order < 2:
            return

        d2h = np.zeros([imgr.N_PX_IMAGE**2,self._nstate,self._nstate])
        for ell1 in range(self._nstate):
            for ell2 in range(ell1+1):
                m_ell_1 = mode_to_phase[:,ell1]
                m_ell_2 = mode_to_phase[:,ell2]
                tmp = 2*((-1j*2*np.pi/wavelength)**2*(
                            (dft_matrix@(g0*m_ell_1*m_ell_2)).conj()*(dft_matrix@g0)
                            - (dft_matrix@(g0*m_ell_1)).conj()*(dft_matrix@(g0*m_ell_2))
                        )).real
                d2h[:,ell1,ell2] = tmp
                d2h[:,ell2,ell1] = tmp
        self.set_dny(d2h,2)
        if self._order < 3:
            return
            
        d3h = np.zeros([imgr.N_PX_IMAGE**2,self._nstate,self._nstate,self._nstate])
        for ell1 in range(self._nstate):
            for ell2 in range(ell1+1):
                for ell3 in range(ell2+1):
                    m_ell_1 = mode_to_phase[:,ell1]
                    m_ell_2 = mode_to_phase[:,ell2]
                    m_ell_3 = mode_to_phase[:,ell3]
                    tmp = 2*((-1j*2*np.pi/wavelength)**3*(
                            (dft_matrix@(g0*m_ell_1*m_ell_2*m_ell_3)).conj()*(dft_matrix@g0)
                            - (dft_matrix@(g0*m_ell_1*m_ell_2)).conj()*(dft_matrix@(g0*m_ell_3))
                            - (dft_matrix@(g0*m_ell_1*m_ell_3)).conj()*(dft_matrix@(g0*m_ell_2))
                            - (dft_matrix@(g0*m_ell_2*m_ell_3)).conj()*(dft_matrix@(g0*m_ell_1))
                        )).real
                    d3h[:,ell1,ell2,ell3] = tmp
                    d3h[:,ell1,ell3,ell2] = tmp
                    d3h[:,ell2,ell1,ell3] = tmp
                    d3h[:,ell2,ell3,ell1] = tmp
                    d3h[:,ell3,ell1,ell2] = tmp
                    d3h[:,ell3,ell2,ell1] = tmp
        self.set_dny(d3h,3)
        if self._order < 4:
            return
        else:
            raise ValueError("maximum taylor order implemented is 3")
    
    def exact_jacobian(self,x):
        if type(x) is np.ndarray:
            CPU = True
            x = cp.array(x)
        elif type(x) is cp.ndarray:
            CPU = False
        else:
            raise TypeError("input x must be either numpy or cupy array")
        b = self._dft_matrix_cp*self.g_cp(x)[None,:]
        out = ((b @ self._mode_to_phase_cp) * (b.conj().sum(axis=1))[:,None]).imag
        out *= -(2*cp.pi/self.wavelength)*2 # not sure why negative here, should be positive surely
        if CPU:
            return out.get()
        else:
            return out
    
    def _exact_jacobian(self,x):
        # deprecated, will be deleted soon
        b = self._dft_matrix*self.g(x)[None,:]
        d1h = ((b @ self._mode_to_phase) * (b.conj().sum(axis=1))[:,None]).imag
        d1h *= -(2*cp.pi/self.wavelength)*2 # not sure why negative here, should be positive surely
        return d1h

    def compass_build_dnys(self,sup,nwfs,get_phase):
        """build dnys for a single WFS using a COMPASS WFS.

        Currently only works for a single full-aperture WFS. Extension to higher
        order WFSs is relatively straightforward, but not implemented.

        Parameters
        ----------
        sup : CompassSupervisor object
            The supervisor object
        nwfs : int
            The index of the WFS to use
        get_phase : function
            A function which returns the phase of the WFS in the 
        """
        offset = np.array(sup.wfs._wfs.d_wfs[nwfs].d_offsets)
        offset -= offset.mean()
        self.general_build_dnys(pup=sup.get_s_pupil(),
            im_width=sup.wfs.get_wfs_image(nwfs).shape[0],
            fft_width=sup.wfs._wfs.d_wfs[nwfs].nfft, get_phase=get_phase,
            offset=-offset,
            wavelength=sup.config.p_wfss[0].get_Lambda())

    def general_build_dnys(self,pup,im_width,fft_width,get_phase,offset,wavelength,
                            delta=1.0):
        """Build the partial derivative arrays for a general simulation.
        
        Parameters
        ----------
        pup : ndarray (pup_width,pup_width)
            The pupil mask.
        im_width : int
            The width of the image.
        fft_width : int
            The width of the fft.
        get_phase : callable
            A function that takes a state vector and returns the phase in the
            pupil plane.
        delta : float
            The step size for the mode-to-phase matrix generation. I.e., if the
            function is a purely linear combination, then delta doesn't matter.
        """

        pup_width = pup.shape[0]
        nstate = self._nstate
        nmeas = self._nmeas

        dft_matrix = la.dft(fft_width,scale=None)
        dft_matrix = np.concatenate([dft_matrix[fft_width//2:,:],dft_matrix[:fft_width//2,]],axis=0)
        dft_matrix = dft_matrix[:,:pup_width]
        dft_matrix = dft_matrix[fft_width//2-im_width//2:fft_width//2+im_width//2,:]
        dft_matrix = np.kron(dft_matrix,dft_matrix)
        dft_matrix = dft_matrix[:,(pup==1).flatten()]

        mode_to_phase = np.zeros([(pup==1).sum(),nstate])

        x = np.zeros(nstate)        
        for i in range(nstate):
            x *= 0.0
            x[i] = delta
            phi = get_phase(x)[pup==1]
            mode_to_phase[:,i] = phi/delta

        self.wavelength = wavelength

        self._offset        = offset[pup==1]
        self._mode_to_phase = mode_to_phase
        self._dft_matrix    = dft_matrix
        self._offset_cp        = cp.array(offset[pup==1])
        self._mode_to_phase_cp = cp.array(mode_to_phase)
        self._dft_matrix_cp    = cp.array(dft_matrix)
        self.g = lambda x: np.exp(1j*(2*np.pi*(self._mode_to_phase@x/self.wavelength)+self._offset))
        self.g_cp = lambda x: cp.exp(1j*(2*cp.pi*(self._mode_to_phase_cp@x/self.wavelength)+self._offset_cp))
        self.h_true = lambda x: np.abs(self._dft_matrix @ self.g(x))**2
        self.h_true_cp = lambda x: cp.abs(self._dft_matrix_cp @ self.g_cp(x))**2
        g0 = self.g(x*0)

        d0h = np.abs(dft_matrix @ g0)**2
        self.set_dny(d0h,0)
        if self._order < 1:
            return
        
        d1h = np.zeros([nmeas,nstate])
        for ell1 in range(nstate):
            m_ell_ = mode_to_phase[:,ell1]
            d1h[:,ell1] = 2*(2*np.pi/self.wavelength)*((-1j)*((dft_matrix@(g0*m_ell_))).conj()*(dft_matrix@g0)).real
        self.set_dny(d1h,1)
        if self._order < 2:
            return

        d2h = np.zeros([nmeas,nstate,nstate])
        for ell1 in range(nstate):
            for ell2 in range(ell1+1):
                m_ell_1 = mode_to_phase[:,ell1]
                m_ell_2 = mode_to_phase[:,ell2]
                tmp = 2*((-1j*2*np.pi/self.wavelength)**2*(
                            (dft_matrix@(g0*m_ell_1*m_ell_2)).conj()*(dft_matrix@g0)
                            - (dft_matrix@(g0*m_ell_1)).conj()*(dft_matrix@(g0*m_ell_2))
                        )).real
                d2h[:,ell1,ell2] = tmp
                d2h[:,ell2,ell1] = tmp
        self.set_dny(d2h,2)
        if self._order < 3:
            return
            
        d3h = np.zeros([nmeas,nstate,nstate,nstate])
        for ell1 in range(nstate):
            for ell2 in range(ell1+1):
                for ell3 in range(ell2+1):
                    m_ell_1 = mode_to_phase[:,ell1]
                    m_ell_2 = mode_to_phase[:,ell2]
                    m_ell_3 = mode_to_phase[:,ell3]
                    tmp = 2*((-1j*2*np.pi/self.wavelength)**3*(
                            (dft_matrix@(g0*m_ell_1*m_ell_2*m_ell_3)).conj()*(dft_matrix@g0)
                            - (dft_matrix@(g0*m_ell_1*m_ell_2)).conj()*(dft_matrix@(g0*m_ell_3))
                            - (dft_matrix@(g0*m_ell_1*m_ell_3)).conj()*(dft_matrix@(g0*m_ell_2))
                            - (dft_matrix@(g0*m_ell_2*m_ell_3)).conj()*(dft_matrix@(g0*m_ell_1))
                        )).real
                    d3h[:,ell1,ell2,ell3] = tmp
                    d3h[:,ell1,ell3,ell2] = tmp
                    d3h[:,ell2,ell1,ell3] = tmp
                    d3h[:,ell2,ell3,ell1] = tmp
                    d3h[:,ell3,ell1,ell2] = tmp
                    d3h[:,ell3,ell2,ell1] = tmp
        self.set_dny(d3h,3)
        if self._order < 4:
            return
        else:
            raise ValueError("maximum taylor order implemented is 3")
    
    def _general_build_dnys(self,pup,im_width,fft_width,get_phase,offset,wavelength,
                            delta=1.0):
        # experimental, do not use. Goal is to build dnys for an arbitrary order,
        # but it's difficult to code up. We have the equations, but the implementation
        # is another thing.
        pup_width = pup.shape[0]
        nstate = self._nstate
        nmeas = self._nmeas

        dft_matrix = la.dft(fft_width,scale=None)
        dft_matrix = np.concatenate([dft_matrix[fft_width//2:,:],dft_matrix[:fft_width//2,]],axis=0)
        dft_matrix = dft_matrix[:,:pup_width]
        dft_matrix = dft_matrix[fft_width//2-im_width//2:fft_width//2+im_width//2,:]
        dft_matrix = np.kron(dft_matrix,dft_matrix)
        dft_matrix = dft_matrix[:,(pup==1).flatten()]

        mode_to_phase = np.zeros([(pup==1).sum(),nstate])

        x = np.zeros(nstate)        
        for i in range(nstate):
            x *= 0.0
            x[i] = delta
            phi = get_phase(x)[pup==1]
            mode_to_phase[:,i] = phi/delta

        self.wavelength = wavelength

        self.g = lambda x: np.exp(1j*(2*np.pi*(mode_to_phase@x/self.wavelength)+offset[pup==1]))
        self.h_true = lambda x: np.abs(self._dft_matrix @ self.g(x))**2
        g0 = self.g(x*0)
        self._mode_to_phase = mode_to_phase
        self._dft_matrix    = dft_matrix

        d0h = np.abs(self.dft_matrix @ g0)**2
        self.set_dny(d0h,0)
        if self._order < 1:
            return
        print("d0 done")
        import time
        t1 = time.time()

        b = self.dft_matrix*g0[None,:]
        d1h = ((b @ self.mode_to_phase) * (b.conj().sum(axis=1))[:,None]).imag
        d1h *= -(2*np.pi/self.wavelength)*2 # not sure why negative here, should be positive surely

        print("d1 done")
        print(f"Time for d1h: {time.time()-t1:0.2f} sec")

        self.set_dny(d1h,1)
        if self._order < 2:
            return
        
        d2h = np.zeros([self._nmeas,self._nstate,self._nstate])
        for ell1 in range(self._nstate):
            for ell2 in range(ell1+1):
                m_ell_1 = mode_to_phase[:,ell1]
                m_ell_2 = mode_to_phase[:,ell2]
                tmp = 2*((-1j*2*np.pi/self.wavelength)**2*(
                            (dft_matrix@(g0*m_ell_1*m_ell_2)).conj()*(dft_matrix@g0)
                            - (dft_matrix@(g0*m_ell_1)).conj()*(dft_matrix@(g0*m_ell_2))
                        )).real
                d2h[:,ell1,ell2] = tmp
                d2h[:,ell2,ell1] = tmp
        self.set_dny(d2h,2)
        if self._order < 3:
            return
        print("d2 done")
            
        d3h = np.zeros([self._nmeas,self._nstate,self._nstate,self._nstate])
        for ell1 in range(self._nstate):
            for ell2 in range(ell1+1):
                for ell3 in range(ell2+1):
                    m_ell_1 = mode_to_phase[:,ell1]
                    m_ell_2 = mode_to_phase[:,ell2]
                    m_ell_3 = mode_to_phase[:,ell3]
                    tmp = 2*((-1j*2*np.pi/self.wavelength)**3*(
                            (dft_matrix@(g0*m_ell_1*m_ell_2*m_ell_3)).conj()*(dft_matrix@g0)
                            - (dft_matrix@(g0*m_ell_1*m_ell_2)).conj()*(dft_matrix@(g0*m_ell_3))
                            - (dft_matrix@(g0*m_ell_1*m_ell_3)).conj()*(dft_matrix@(g0*m_ell_2))
                            - (dft_matrix@(g0*m_ell_2*m_ell_3)).conj()*(dft_matrix@(g0*m_ell_1))
                        )).real
                    d3h[:,ell1,ell2,ell3] = tmp
                    d3h[:,ell1,ell3,ell2] = tmp
                    d3h[:,ell2,ell1,ell3] = tmp
                    d3h[:,ell2,ell3,ell1] = tmp
                    d3h[:,ell3,ell1,ell2] = tmp
                    d3h[:,ell3,ell2,ell1] = tmp
        self.set_dny(d3h,3)
        print("d3 done")
        if self._order < 4:
            return
        else:
            raise ValueError("maximum taylor order implemented is 3")
    