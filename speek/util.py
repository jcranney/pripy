try:
    import cupy as cp
    import cupyx.scipy.linalg as la
except ImportError as ie:
    Warning("No cupy available, using basic numpy/scipy")
    import numpy as cp
    import scipy.linalg as la

import numpy as np

from scipy.special import factorial

class TaylorHModel:
    def __init__(self,order,nmeas,nstate):
        self._order  = order
        self._nmeas  = nmeas
        self._nstate = nstate
        self._einbcd = "bcdefghijklmnopqrstuvwxyz"
        self.dny     = []
        assert order >= 0        
        if order > 5:
            Warning("High order taylor estimate, are you sure?")

        for ni in range(order+1):
            self.dny.append(cp.zeros([nmeas]+[nstate]*ni))
        self._einstring = ["a"+self._einbcd[:ni]+"".join([","+x for x in self._einbcd[:ni]])+"->a" for ni in range(self._order+1)]

    def set_dny(self,dny,n):
        assert dny.shape==self.dny[n].shape
        if type(dny) is cp.ndarray:
            self.dny[n] = dny.copy()
        else:
            self.dny[n][:] = cp.array(dny)
    
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
            out += (1/factorial(ni))*cp.einsum(self._einstring[ni],self.dny[ni],*([x]*ni))
        if CPU:
            return out.get()
        else:
            return out

    def ceo_build_dnys(self,gmt,src,imgr):
        n_px_fft = imgr.DFT_osf*imgr.N_PX_PUPIL        
        pup = src.amplitude.host()
        xx,yy = cp.meshgrid(cp.arange(pup.shape[0]),cp.arange(pup.shape[0]))

        pup_mask = pup==1

        phi = pup.copy()

        dft_matrix = la.dft(n_px_fft)
        dft_matrix = cp.concatenate([dft_matrix[n_px_fft//2:,:],dft_matrix[:n_px_fft//2,]],axis=0)
        dft_matrix = dft_matrix[:,:imgr.N_PX_PUPIL]
        dft_matrix = dft_matrix[(n_px_fft-imgr.N_PX_IMAGE)//2+1:(n_px_fft+imgr.N_PX_IMAGE)//2+1,:]
        dft_matrix = cp.kron(dft_matrix,dft_matrix)
        dft_matrix = dft_matrix[:,pup_mask.flatten()]

        ~gmt

        mode_to_phase = cp.zeros([pup_mask.sum(),self._nstate])

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

        wavelength = src.wavelength*1e6
            
        g0 = cp.exp((1j*2*cp.pi*(xx+yy)/pup.shape[0]/4))[pup_mask]

        d0h = cp.abs(dft_matrix @ g0)**2
        self.set_dny(d0h,0)
        if self._order < 1:
            return
        
        d1h = cp.zeros([imgr.N_PX_IMAGE**2,self._nstate])
        for ell1 in range(self._nstate):
            m_ell_ = mode_to_phase[:,ell1]
            d1h[:,ell1] = 2*(2*cp.pi/wavelength)*((-1j)*((dft_matrix@(g0*m_ell_))).conj()*(dft_matrix@g0)).real
        self.set_dny(d1h,1)
        if self._order < 2:
            return

        d2h = cp.zeros([imgr.N_PX_IMAGE**2,self._nstate,self._nstate])
        for ell1 in range(self._nstate):
            for ell2 in range(ell1+1):
                m_ell_1 = mode_to_phase[:,ell1]
                m_ell_2 = mode_to_phase[:,ell2]
                tmp = 2*((-1j*2*cp.pi/wavelength)**2*(
                            (dft_matrix@(g0*m_ell_1*m_ell_2)).conj()*(dft_matrix@g0)
                            - (dft_matrix@(g0*m_ell_1)).conj()*(dft_matrix@(g0*m_ell_2))
                        )).real
                d2h[:,ell1,ell2] = tmp
                d2h[:,ell2,ell1] = tmp
        self.set_dny(d2h,2)
        if self._order < 3:
            return
            
        d3h = cp.zeros([imgr.N_PX_IMAGE**2,self._nstate,self._nstate,self._nstate])
        for ell1 in range(self._nstate):
            for ell2 in range(ell1+1):
                for ell3 in range(ell2+1):
                    m_ell_1 = mode_to_phase[:,ell1]
                    m_ell_2 = mode_to_phase[:,ell2]
                    m_ell_3 = mode_to_phase[:,ell3]
                    tmp = 2*((-1j*2*cp.pi/wavelength)**3*(
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