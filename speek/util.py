try:
    import cupy as cp
except ImportError as ie:
    Warning("No cupy available, using basic numpy/scipy")
    import numpy as cp

import numpy as np

from scipy.special import factorial
import scipy.linalg as la

class FastAndFurious:
    """ Fast and Furious is a class for computing the phase of a wavefront
    from an image using sequential phase diversity. It is based on the
    algorithm described in Korkiakoski et al, Appl. Opt. 53, 4565-4579 (2014).

    Parameters
    ----------
    pup_small : ndarray
        The pupil function.
    im_width : int
        The width of the image.
    fft_width : int
        The width of the FFT.
    offset : ndarray (pup_width x pup_width)
        The offset of the wavefront phase (in radians) required to (e.g.) centre
        the focal plane image over 2x2 pixels (rather than centred over 1x1).
    epsilon : float/ndarray (fft_width x fft_width)
        The regularisation parameter used in the focal plane of the FF algorithm. 
        Can be a scalar or an array in the fft domain. E.g., this parameter
        should be set to a large number if the corresponding pixel is noisy/unused.
        See example for details.

    Attributes
    ----------
    pup_width : int
        The width of the pupil plane arrays.
    im_width : int
        The width of the focal plane arrays.
    fft_width : int 
        The width of the FFT support.
    
    Methods
    -------
    get_phase(p_i)
        Compute the phase of the wavefront from an image.
    set_diversity_phase(dphi)
        Set the diversity phase.
    """
    def __init__(self, pup: np.ndarray, im_width: int, fft_width: int, 
                offset: np.ndarray, epsilon: float = 1e-5):
        """ Initialise.
        """
        self.im_width   = im_width           # width of image
        self.pup_width  = pup.shape[0] # width of the pupil
        self.fft_width  = fft_width          # width of fft

        # Set up the (hidden) parameters. These are all either scalars or cupy
        # arrays.
        # Pupil in FFT dimensions:
        self._pup = cp.pad(cp.array(pup),(self.fft_width-pup.shape[0])//2) 
        # Offset for (e.g.) centering the image on 2x2 pixels:
        self._offset = self._pup * cp.exp(1j*cp.pad(cp.array(offset),(self.fft_width-self.pup_width)//2))
        # Recursive parameters:
        self._v_d = None
        self._y_d = None
        self._p_de = None
        # Reset flag:
        self._reset = True
        # Focal plane complex amplitude (real valued -- assuming symmetry in pupil):
        self._a = (cp.fft.fftshift(cp.fft.fft2(self._pup*self._offset))).real
        self._a /= self._pup.sum()
        # Focal plane intensity:
        self._a2 = self._a**2
        # Regularisation parameter:
        self._epsilon = cp.array(epsilon)
        # Constants:
        self._ydenom = 2*self._a**2+self._epsilon
        self._S = 1 # TODO: estimate this on the fly      

    def reset(self):
        """ Reset the FF algorithm/recursive parameters
        """
        self._v_d   = None
        self._y_d   = None
        self._p_de  = None
        self._reset = True

    def retrieve_phase(self,p_i):
        """ Compute the phase of the wavefront from an image.

        Note that the phase in all variables within FF is expected to be in
        radians, so typically one will need to scale returned phase by 
        lambda/(2*pi) after calling this function.

        Parameters
        ----------
        p_i : ndarray (im_width x im_width)
            The image used for wavefront retrieval.

        Returns
        -------
        ndarray (pup_width x pup_width)
            The estimated phase of the wavefront.
        """

        # rescale the image and pad to the fft size:
        p_i = cp.pad(cp.array(p_i),(self.fft_width-self.im_width)//2)
        p_i *= self._a2.sum() / p_i.sum()
        p_i += (1-p_i.max()/self._a2.max())*self._a2

        # odd/even decomposition:
        p_io = 0.5*(p_i-p_i[::-1,::-1]) # odd component of image
        p_ie = 0.5*(p_i+p_i[::-1,::-1]) # even component of image
        
        # compute imaginary part of FFT(phase):
        y_i = self._a*p_io/self._ydenom
        
        # compute sign of real part of FFT(phase):
        if self._reset:
            v_sign = cp.sign(self._a.copy())
            self._reset = False
        else:
            v_sign = cp.sign((self._p_de-p_ie-(self._v_d**2+self._y_d**2+2*y_i*self._y_d))/(2*self._v_d))
        
        # compute abs(real part of FFT(phase)):
        v_i_abs = cp.sqrt(cp.abs(p_ie-(self._S*self._a2+y_i**2)))

        # combine to get real part of FFT(phase):
        v_i = v_i_abs * v_sign

        # inverse transform to get phase + negative because I might have a sign 
        # error somewhere else?:
        phi = - cp.fft.ifft2(cp.fft.fftshift(v_i+1j*y_i),norm="forward").real
        
        # save even part of image for next iteration:
        self._p_de = p_ie.copy()

        # return phase estimate on pupil plane support:
        phi =  phi[self.fft_width//2-self.pup_width//2:self.fft_width//2+self.pup_width//2,
                   self.fft_width//2-self.pup_width//2:self.fft_width//2+self.pup_width//2]
        # return numpy array regardless of if using cupy or not:
        if hasattr(phi,"get"):
            return phi.get()
        else:
            return phi

    def set_diversity_phase(self,dphi):
        """ Set the diversity phase. This should be the difference between the 
        most recent phase and the phase of the wavefront at the previous image.
        I.e., the difference between the two DM shapes, assuming the turbulence
        change is negligble.

        Note that the phase in all variables within FF is expected to be in
        radians, so typically one will need to scale the phase by 2*pi/lambda 
        before passing to this function.

        Parameters
        ----------
        dphi : ndarray (pup_width x pup_width)
            The diversity phase. 
        """
        dphi_big = cp.pad(cp.array(dphi),(self.fft_width-self.pup_width)//2)
        dphi_big_o = (dphi_big-dphi_big[::-1,::-1])*0.5
        dphi_big_e = (dphi_big+dphi_big[::-1,::-1])*0.5
        self._y_d = cp.fft.fftshift(cp.fft.ifft2(dphi_big_o)).imag
        self._v_d = cp.fft.fftshift(cp.fft.ifft2(dphi_big_e)).real



class GerchbergSaxton:
    def __init__(self,pup,wavelength,fft_width,im_width):
        self.im_width = im_width
        self.fft_width = fft_width
        self.pup = cp.pad(cp.array(pup),(self.fft_width-pup.shape[0])//2)
        self.pup_shft = cp.fft.fftshift(self.pup)
        self.wavelength = wavelength
        self.scf = (cp.abs(cp.fft.fft2(self.pup_shft))**2).sum()
        self.half_pix_phase = cp.mgrid[:self.fft_width,:self.fft_width].sum(axis=0)*(cp.pi/self.fft_width)
        self.half_pix_phase = self.half_pix_phase[self.pup==1]
        self.invalid_pixels = cp.fft.fftshift(cp.pad(cp.ones([im_width,im_width]),(self.fft_width-im_width)//2)==0)

    def rebin(self, a, newshape ):
            '''Rebin an array to a new shape.
            newshape must be a factor of a.shape.
            '''
            slices = [ slice(None,None, old/new) for old,new in zip(a.shape,newshape) ]
            return a[slices]

    def gs_run(self,phasepup_shft,amp_shft,iterations=20,hio_param=0.3):
        # inputs:
        # ~~~~~~~
        #
        # phasepup_shft: starting phase at the pupil plane
        # pupil_shft: amplitude at the pupil plane
        # amp_shft: amplitude at the focal plane
        # iterations: number of iterations
        #
        # outputs:
        # ~~~~~~~~
        #
        # phasepup_shft: final phase at the pupil plane
        #
        amppup_shft = self.pup_shft.copy()
        for it in range(iterations):
            cplxpup_shft = ((1.+hio_param)*self.pup_shft-hio_param*amppup_shft)*cp.exp(-1j*phasepup_shft)    
            cplxim_shft = cp.fft.fft2(cplxpup_shft)
            
            phaseim_shft = -cp.angle(cplxim_shft)
            ampim_shft = cp.abs(cplxim_shft)
            
            cplxim_shft2 = ((1.+hio_param)*amp_shft-hio_param*ampim_shft)*cp.exp(-1j*phaseim_shft)
            if self.invalid_pixels is not None:
                cplxim_shft2[self.invalid_pixels] = cplxim_shft[self.invalid_pixels].copy() 

            cplxpup_shft2 = cp.fft.ifft2(cplxim_shft2)
            phasepup_shft = -cp.angle(cplxpup_shft2)
            amppup_shft = cp.abs(cplxpup_shft2)
            
        return phasepup_shft

    def get_phase(self,im,iters=100,init=None,hio_param=0.3):
        im = cp.array(im)*(self.scf/im.sum())
        im = cp.pad(im,(self.fft_width-self.im_width)//2)
        im = im**0.5
        im_shft = cp.fft.fftshift(im)
        
        if init is None:
            init = cp.random.randn(*self.pup_shft.shape)/10
        else:
            if cp.all(cp.r_[init.shape]==self.fft_width):
                pass
            else:
                init = cp.pad(init,(self.fft_width-init.shape[0])//2)

        out_phi_shft = self.gs_run(init,im_shft,iterations=iters,hio_param=hio_param)
        out_phi  = cp.fft.ifftshift(out_phi_shft)[self.pup==1]
        out_phi -= self.half_pix_phase
        out_phi *= self.wavelength/(2*cp.pi)
        out_phi -= out_phi.mean()
        return out_phi

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
    
    def prep(self,sup,nwfs,update_state):    
        n_px_fft = sup.get_i_pupil().shape[0]
        n_px_image = sup.wfs.get_wfs_image(nwfs).shape[0]
        pup = np.array(sup.wfs._wfs.d_wfs[nwfs].d_pupil)
        xx,yy = cp.meshgrid(cp.arange(pup.shape[0]),cp.arange(pup.shape[0]))
        pup_mask = pup==1
        phi = pup.copy()
        dft_matrix = la.dft(n_px_fft,scale="sqrtn")
        dft_matrix = np.concatenate([dft_matrix[n_px_fft//2:,:],dft_matrix[:n_px_fft//2,]],axis=0)
        dft_matrix = dft_matrix[:,:pup.shape[0]]
        dft_matrix = dft_matrix[n_px_fft//2-n_px_image//2+1:n_px_fft//2+n_px_image//2+1,:]
        dft_matrix = np.kron(dft_matrix,dft_matrix)
        dft_matrix = dft_matrix[:,pup_mask.flatten()]
        mode_to_phase = np.zeros([pup_mask.sum(),self._nstate])
        x = np.zeros(self._nstate)        
        for i in range(self._nstate):
            x *= 0.0
            delta = 1.0
            x[i] = delta
            update_state(x)
            phi = sup.wfs.get_wfs_phase(nwfs)*pup
            mode_to_phase[:,i] = phi[pup_mask]/delta
        self.wavelength = sup.config.p_wfss[nwfs].Lambda
        self.h_true = lambda x: cp.abs(self.dft_matrix @ self.g(x))**2
        self.mode_to_phase = cp.array(mode_to_phase)
        self.dft_matrix    = cp.array(dft_matrix)
        self.g = lambda x: cp.exp(1j*2*cp.pi*(self.mode_to_phase@x/self.wavelength+(xx+yy)[pup_mask]/n_px_fft/2))



    def compass_build_dnys_general(self,sup,nwfs,update_state):
        # nwfs for now is just the index of the WFS we want to use (and it must
        # be a 1x1 for now)
        # update_state is a function called with a vector argument which updates
        # the wfs phase to be the appropriate state.
        if self.g is None:
            self.prep(sup,nwfs,update_state)

        x = np.zeros(self._nstate)        
        x0 = x*0
        g0 = self.g(x0)

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
        
        d2h = np.zeros([n_px_image**2,self._nstate,self._nstate])
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
            
        d3h = np.zeros([n_px_image**2,self._nstate,self._nstate,self._nstate])
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
        if self._order < 4:
            return
        else:
            raise ValueError("maximum taylor order implemented is 3")
        print("d3 done")
    
    def exact_jacobian(self,x):
        b = self.dft_matrix*self.g(x)[None,:]
        d1h = ((b @ self.mode_to_phase) * (b.conj().sum(axis=1))[:,None]).imag
        d1h *= -(2*np.pi/self.wavelength)*2 # not sure why negative here, should be positive surely
        return d1h

    def compass_build_dnys(self,sup,nwfs,update_state):
        # nwfs for now is just the index of the WFS we want to use (and it must
        # be a 1x1 for now)
        # update_state is a function called with a vector argument which updates
        # the wfs phase to be the appropriate state.

        n_px_fft = sup.get_i_pupil().shape[0]
        n_px_image = sup.wfs.get_wfs_image(nwfs).shape[0]
        pup = np.array(sup.wfs._wfs.d_wfs[nwfs].d_pupil)
        xx,yy = np.meshgrid(np.arange(pup.shape[0]),np.arange(pup.shape[0]))

        pup_mask = pup==1

        print(f"{n_px_fft=}")
        print(f"{n_px_image=}")
        phi = pup.copy()
        dft_matrix = la.dft(n_px_fft,scale="sqrtn")
        dft_matrix = np.concatenate([dft_matrix[n_px_fft//2:,:],dft_matrix[:n_px_fft//2,]],axis=0)
        dft_matrix = dft_matrix[:,:pup.shape[0]]
        dft_matrix = dft_matrix[n_px_fft//2-n_px_image//2+1:n_px_fft//2+n_px_image//2+1,:]
        print(dft_matrix.shape)
        dft_matrix = np.kron(dft_matrix,dft_matrix)
        dft_matrix = dft_matrix[:,pup_mask.flatten()]

    

        mode_to_phase = np.zeros([pup_mask.sum(),self._nstate])

        x = np.zeros(self._nstate)        
        for i in range(self._nstate):
            x *= 0.0
            delta = 1.0
            x[i] = delta
            update_state(x)
            phi = sup.wfs.get_wfs_phase(nwfs)*pup
            mode_to_phase[:,i] = phi[pup_mask]/delta

        self.wavelength = sup.config.p_wfss[nwfs].Lambda
        print(self.wavelength)

        #g0 = np.exp((xx+yy)/pup.shape[0]/2)[pup_mask]
        self.g = lambda x: np.exp(1j*2*np.pi*(mode_to_phase@x/self.wavelength+(xx+yy)[pup_mask]/n_px_fft/2))
        self.h_true = lambda x: np.abs(self._dft_matrix @ self.g(x))**2
        g0 = self.g(x*0)
        self._mode_to_phase = mode_to_phase
        self._dft_matrix    = dft_matrix

        d0h = np.abs(dft_matrix @ g0)**2
        self.set_dny(d0h,0)
        if self._order < 1:
            return
        
        d1h = np.zeros([n_px_image**2,self._nstate])
        for ell1 in range(self._nstate):
            m_ell_ = mode_to_phase[:,ell1]
            d1h[:,ell1] = 2*(2*np.pi/self.wavelength)*((-1j)*((dft_matrix@(g0*m_ell_))).conj()*(dft_matrix@g0)).real
        self.set_dny(d1h,1)
        if self._order < 2:
            return

        d2h = np.zeros([n_px_image**2,self._nstate,self._nstate])
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
            
        d3h = np.zeros([n_px_image**2,self._nstate,self._nstate,self._nstate])
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
        if self._order < 4:
            return
        else:
            raise ValueError("maximum taylor order implemented is 3")

#TODO: make a class for an object that can be initialised like the taylor class above
#      but instead allows the analytical evaluation, jacobian, and hessian to be computed
#      arbitrarily and as efficiently as possible.