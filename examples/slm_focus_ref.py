# -*- coding: utf-8 -*-
"""
Usage: test_slm.py [options]

    -s, --slm               Use the SLM
    -m, --monitor <id>      The monitor to use, -1 means virtual SLM [default: -1]
"""

from docopt import docopt
import cv2
import numpy as np
import slmpy
from math import factorial
import torch as t
t.set_num_threads(6)
from pointgrey_handler import CameraHandler as CameraHandler
import time
import matplotlib.pyplot as plt
from slm_make_gmt_pupil import make_phimat,make_pupil,make_tt_phimat,make_foc_phimat
from tqdm import tqdm
from scipy.signal import fftconvolve

plt.ion()

def rebin(a, factor):
    sh = a.shape[0]//factor,factor,a.shape[1]//factor,factor
    return a.reshape(sh).mean(-1).mean(1)

class Controller:
    
    init = True            # True when requiring initialisation
    calibration_status = 0 # zero when finished or not yet started

    def __init__(self, im_width: int, pup_width: int, fft_width: int,
                rebin_factor: int, nstate: int, pup_s: np.ndarray, 
                get_phase: callable):
        self.pup_width = pup_width
        self.im_width = im_width
        self.fft_width = fft_width
        self.rebin_factor = rebin_factor
        self.pup_s = t.tensor(pup_s,dtype=t.float32)
        yy_s,xx_s = t.meshgrid(t.arange(pup_width)/(pup_width-1)-0.5,
                               t.arange(pup_width)/(pup_width-1)-0.5)
    
        # offset to get the image centred on 2x2 pixels
        if self.im_width % 2 == 0:
            self.phase_offset = -(xx_s+yy_s)*(pup_width-1)*2*t.pi/fft_width/2
        else:
            self.phase_offset = xx_s*0.0
        self.get_phase = get_phase
        self.nstate = nstate
        self.nmeas  = im_width**2

    def phase_to_image(self,phi):
        """Take wavefront phase in small pupil dimensions and return image
        
        Args:
            phi (ndarray): phase in small pupil dimensions

        Returns:
            ndarray: focal plane image
        """
        wf_s = self.pup_s * t.exp(1j*(phi+self.phase_offset))
        im_F = t.abs(t.fft.fftshift(t.fft.fft2(wf_s,s=[self.fft_width,self.fft_width]))/self.pup_s.sum())**2
        return im_F

    @staticmethod
    def rebin(a, factor):
        sh = a.shape[0]//factor,factor,a.shape[1]//factor,factor
        return a.reshape(sh).mean(-1).mean(1)

    def trim_im(self,im):
        """Trim image to a square of size trimmed_width x trimmed_width

        Args:
            im (ndarray): image to trim
            trimmed_width (int): size of the trimmed image

        Returns:
            ndarray: trimmed image
        """
        im = rebin(im,self.rebin_factor)
        if self.im_width % 2 == 0: # even
            return im[
                im.shape[0]//2-self.im_width//2:im.shape[0]//2+self.im_width//2,
                im.shape[1]//2-self.im_width//2:im.shape[1]//2+self.im_width//2
                ]
        else: # odd
            return im[
                im.shape[0]//2-self.im_width//2:im.shape[0]//2+self.im_width//2+1,
                im.shape[1]//2-self.im_width//2:im.shape[1]//2+self.im_width//2+1
                ]

    def h_func(self,x):
        "takes state, returns image"
        return self.trim_im(self.phase_to_image(self.get_phase(x)))

    def init_mhe(self,nbuffer,sigma_x,sigma_w):
        self.nbuffer = nbuffer
        self.Sigma_x_inv_fact = 1/(sigma_x)**0.5*t.eye(self.nstate)
        self.Sigma_w_inv_fact = 1/(sigma_w)**0.5*t.eye(self.nmeas)
        
        self.x = t.zeros([self.nstate],requires_grad=True)
        self.optimizer = t.optim.SGD([self.x],lr=0.01)

        self.gain = 0.3
        self.leak = 0.995

        self.x_dm = t.zeros([self.nbuffer,self.nstate])
        self.yd = t.zeros([self.nbuffer,self.nmeas])
        self.x_corr = t.zeros(self.nstate)

        self.iter = 0

    def cost_ls(self,x,u,y):
        z = t.zeros(self.nbuffer*self.nmeas)
        for n in range(self.nbuffer):
            tmp = y[n]-self.h_func(x+u[n])[self.im_mask==1]
            z[n*self.nmeas:(n+1)*self.nmeas] = t.einsum("i,ij->j",tmp,self.Sigma_w_inv_fact)
        return t.concat([z,1e0*x],dim=0)
        #return z

    def update(self,frame):
        self.yd   = self.yd.roll(-1,0)
        self.x_dm = self.x_dm.roll(-1,0)
        self.yd[-1,:]   = t.tensor(frame.flatten()).to(t.float32)
        self.x_dm[-1,:] = self.x_corr
        #with t.no_grad():
        #    self.x *= 0.0
        if self.iter >= self.nbuffer:
            mu = 0.001
            nu = 2.0
            ep_1 = ep_2 = 1e-6
            jac = t.autograd.functional.jacobian(lambda x : self.cost_ls(x,self.x_dm,self.yd),self.x,vectorize=True,strategy="forward-mode")
            f_x = self.cost_ls(self.x,self.x_dm,self.yd)
            old_cost = (f_x**2).sum()
            for _ in tqdm(range(20),leave=False):
                dx  = -t.linalg.solve(jac.T @ jac + mu * t.eye(self.nstate), jac.T @ f_x)
                if t.norm(dx,"fro") <= ep_2*(t.norm(self.x,"fro")+ep_2):
                    break
                x_new = self.x + dx
                cost = (self.cost_ls(x_new,self.x_dm,self.yd)**2).sum()
                rho = (old_cost - cost)/(0.5*dx @ (mu*dx - jac.T @ f_x))
                if rho > 0:
                    with t.no_grad():
                        self.x += dx
                    jac = t.autograd.functional.jacobian(lambda x : self.cost_ls(x,self.x_dm,self.yd),self.x,vectorize=True,strategy="forward-mode")
                    f_x = self.cost_ls(self.x,self.x_dm,self.yd)
                    old_cost = (f_x**2).sum()
                    mu = mu * max(1/3,1-(2*rho-1)**3)
                    nu   = 2
                else:
                    mu = mu * nu
                    nu = 2 * nu
            xk_opt = self.x.clone().detach().numpy()
            self.x_corr    = self.leak*(1-self.gain)*self.x_corr - self.gain*(xk_opt)
        self.iter += 1

class SLM(slmpy.SLMdisplay):
    """Class to extend the functionality of SLMdisplay to
    include configuration parameters and standard functions
    """

    def __init__(self,*args,monitor=1,wrap=False,x0=0,y0=0,
        mini_res_x=800,mini_res_y=600,mask=None,phimat=None,**kwargs):
        """initialise SLM object
        
        Args:
            wrap (bool) : if True, wrap phase to be between 0 and 1, otherwise saturate.
            x0 (float) : x-offset of coordinate system in pixels
            y0 (float) : y-offset of coordinate system in pixels
        
        see also: slmpy.SLMdisplay
        """
        self._mini_res_x, self._mini_res_y = (mini_res_x, mini_res_y)
        if monitor >= 0:
            # Real SLM, initialise super class and set resolution accordingly
            self._virtual = False
            super().__init__(*args,monitor=monitor,**kwargs)
            self._res_x, self._res_y = self.getSize()
        else:
            # Virtual SLM, set resolution to mini dimensions
            self._virtual = True
            self._res_x, self._res_y = (mini_res_x, mini_res_y)
        self._wrap  = wrap
        yy,xx = np.mgrid[:self._res_y,:self._res_x]*1.0
        xx -= x0
        yy -= y0
        self._xx = xx/self._res_y
        self._yy = yy/self._res_y
        self._image8bit = np.zeros([self._res_y,self._res_x],dtype=np.uint8)
        self._image = np.zeros([self._res_y,self._res_x],dtype=np.float64)
        if mask is None:
            self._mask = np.ones([self._res_y,self._res_x],dtype=np.float64)
        else:
            self._mask = mask.astype(np.float64)
        self._phimat = phimat
    
    def update(self,wrap=None):
        """normalise (including wrap), convert to 8bit, and apply to SLM

        Args:
            wrap (bool) : force wrapping of phase, if None then use SLM default.
        """
        if wrap is None:
            wrap = self._wrap
        
        # wrap image if required
        if wrap:
            self._image %= 1.0
        
        # clip image to [0,1]
        self._image[self._image<0] = 0
        self._image[self._image>1] = 1
        
        # convert to 8bit
        self._image8bit[...] = np.round(self._image*255).astype(np.uint8)

        if not self._virtual:
            # send to SLM
            super().updateArray(self._image8bit,sleep=0.01)

        self._update_mini(self._image8bit)

    def _update_mini(self,slm_image8bit):
        # convert image8bit to mini display format
        slm_image = cv2.resize(slm_image8bit,(self._mini_res_x,self._mini_res_y), interpolation = cv2.INTER_LINEAR)
        slm_image = cv2.cvtColor(slm_image, cv2.COLOR_GRAY2BGR)
        # display image on window
        cv2.imshow('Phase mask',slm_image)
        cv2.waitKey(1)

    def clear_phase(self,update=False):
        self._image *= 0.0
        if update:
            self.update()
    
    def set_phimat(self,phimat):
        self._phimat = phimat

    def add_phimat_phase(self,x,update=False):
        if self._phimat is None:
            raise ValueError("phimat has not been set")
        self._image += np.einsum("ijk,i->jk",self._phimat,x)
        if update:
            self.update()

    def add_piston_phase(self,z,update=False):
        self._image += z
        if update:
            self.update()
    
    def add_poly(self,nx,ny,a,update=False):
        self._image += self._xx**nx * self._yy**ny * a / factorial(nx+ny)
        if update:
            self.update()
    
    def apply_mask(self,mask=None,update=False):
        if mask is None:
            mask = self._mask
        self._image *= mask
        if update:
            self.update()
    
    def set_mask(self,mask):
        self._mask = mask.astype(np.float64)

if __name__ == "__main__":
    doc = docopt(__doc__)

    # parameters (eventually should be moved to cli/docopt)
    # global params
    wavelength = 0.633

    # controller params
    ctrl_im_width_full = 40
    ctrl_im_rebin = 2

    ctrl_im_width = ctrl_im_width_full // ctrl_im_rebin
    ctrl_pup_width = 100
    ctrl_pup_kwargs = {
        "x0" : ctrl_pup_width/2,
        "y0" : ctrl_pup_width/2,
        "x_res" : ctrl_pup_width,
        "y_res" : ctrl_pup_width,
        "seg_diam" : ctrl_pup_width/(3+0.359/8.4*2)
    }
    ctrl_pup_s = t.tensor(make_pupil(**ctrl_pup_kwargs)).to(t.float32)
    ctrl_phimat = t.tensor(np.concatenate([
        make_tt_phimat(**ctrl_pup_kwargs),
        make_foc_phimat(**ctrl_pup_kwargs),
        ],axis=0)).to(t.float32)
    ctrl_fft_width = 384

    # slm params
    slm_monitor_id = int(doc["--monitor"])
    slm_full_width = 556 # full width of GMT pupil across long axis
    slm_x0 = 438
    slm_y0 = 302
    slm_seg_diam = slm_full_width/(3+0.359/8.4*2) # segment diameter in pixels
    

    # camera params
    cam_im_width_full = 80
    cam_im_rebin = 4
    assert(cam_im_width_full % cam_im_rebin == 0)
    cam_im_width = cam_im_width_full // cam_im_rebin
    cam_offset_x = 464
    cam_offset_y = 550
    cam_pixel_as = 1.0 # TODO compute this from physical system parameters
    cam_dark_frame = np.load("dark.npy")

    # initialise controller
    ctrl = Controller(im_width=ctrl_im_width, pup_width=ctrl_pup_width,
                fft_width=ctrl_fft_width, rebin_factor=ctrl_im_rebin,
                nstate=ctrl_phimat.shape[0], pup_s=ctrl_pup_s,
                get_phase = lambda x : t.einsum("ijk,i->jk",ctrl_phimat,x))

    # initialise slm
    slm = SLM(monitor=slm_monitor_id,isImageLock=True,wrap=True)
    slm_phimat = np.concatenate([
        make_tt_phimat(x0=slm_x0, y0=slm_y0, 
                x_res=slm._res_x, y_res=slm._res_y,
                seg_diam=slm_seg_diam),
        make_foc_phimat(x0=slm_x0, y0=slm_y0, 
                x_res=slm._res_x, y_res=slm._res_y,
                seg_diam=slm_seg_diam),
        ],axis=0)
    slm.set_phimat(phimat=slm_phimat*0.14) # this constant is related to wavelength
    slm.clear_phase()
    slm.update()

    # initialise camera
    cam  = CameraHandler(width=cam_im_width_full, height=cam_im_width_full,
                offset_x=cam_offset_x, offset_y=cam_offset_y,
                exposure=100e3, gain=5.0)
    u = np.zeros(ctrl_phimat.shape[0])
    # run tt loop to find offset
    yy,xx = np.mgrid[:cam_im_width,:cam_im_width]-cam_im_width/2+0.5
    if 1==0:
        with cam.FrameGrabber(cam) as fg:
            fig,ax = plt.subplots(figsize=[3,3])
            ax.imshow(rebin(fg.grab(1)[0].astype(np.float32),cam_im_rebin))
            ax.images[0].set_clim(0,2**16)
            for _ in range(100):
                # grab a frame
                frame = rebin(fg.grab(1)[0]*1.0,cam_im_rebin)-cam_dark_frame
                frame = frame.astype(np.float32)
                frame[frame<0] = 0
                ax.images[0].set_data(frame)
                # update command
                cogx = frame.flatten() @ xx.flatten() / frame.sum()
                cogy = frame.flatten() @ yy.flatten() / frame.sum()
                s = np.r_[cogx,cogy]
                u[:2] = u[:2] - 0.2 * s
                
                # apply command
                slm.clear_phase()
                slm.add_piston_phase(0.5)
                slm.add_phimat_phase(u)
                slm.update()

                # wait a bit
                cv2.waitKey(20)
        print(u)
    else:       
        u[0] = 0.5
        u[1] = 8.5
    
    u_ref = u.copy()
    if 1==0:
        # focus:
        u = u_ref.copy()
        intsty = []
        fs = np.linspace(-2,2,21)
        slm.clear_phase()
        slm.add_piston_phase(0.5)
        slm.add_phimat_phase(u)
        slm.update()
        cv2.waitKey(100)
        s = []
        with cam.FrameGrabber(cam) as fg:
            fig,ax = plt.subplots(figsize=[3,3])
            ax.imshow(rebin(fg.grab(1)[0].astype(np.float32),cam_im_rebin))
            ax.images[0].set_clim(0,2**16)
            for f in fs:
                # update command
                u[2] = f
                
                # apply command
                slm.clear_phase()
                slm.add_piston_phase(0.5)
                slm.add_phimat_phase(u)
                slm.update()

                # wait a bit
                cv2.waitKey(20)
                
                # grab a frame
                frame = rebin(np.max(fg.grab(50),axis=0),cam_im_rebin)-cam_dark_frame
                frame = frame.astype(np.float32)
                cogx = frame.flatten() @ xx.flatten() / frame.sum()
                cogy = frame.flatten() @ yy.flatten() / frame.sum()
                s.append(np.r_[cogx,cogy])
                intsty.append(frame.max())
                ax.images[0].set_data(frame)
        cv2.waitKey(100)
        s = np.array(s)
        """
        plt.figure()
        plt.plot(fs,s[:,0],label="tip")
        plt.plot(fs,s[:,1],label="tilt")
        plt.xlabel("defocus")
        plt.ylabel("slope")
        plt.axis("square")
        plt.xlim([-5,5])
        plt.ylim([-5,5])
        cv2.waitKey(100)
        """
        intsty = np.array(intsty)
        f_ref = fs[np.argmax(intsty)]
        u_ref[2] = f_ref
        plt.figure()
        plt.plot(fs,intsty)
    else:
        u_ref[2] = -0.5
    print(u_ref)



    # calibrate
    # compare analytical and acquired image with no phase applied
    x_test = np.zeros(ctrl_phimat.shape[0])
    y_test_ctrl = ctrl.h_func(t.tensor(x_test).to(t.float32)).detach().numpy()
    slm.clear_phase()
    slm.add_piston_phase(0.5)
    slm.add_phimat_phase(u_ref)
    slm.add_phimat_phase(x_test)
    slm.update()
    cv2.waitKey(100)
    with cam.FrameGrabber(cam) as fg:
        y_test_cam = rebin(np.mean(fg.grab(10),axis=0),cam_im_rebin)-cam_dark_frame
    flux_ratio = y_test_ctrl.std()/y_test_cam.std()
    plt.matshow(y_test_ctrl)
    plt.colorbar()
    plt.matshow(y_test_cam*flux_ratio)
    plt.colorbar()
    plt.matshow(y_test_ctrl-y_test_cam*flux_ratio)
    plt.colorbar()
    plt.title("difference no aberrations")
    cv2.waitKey(100)

    im_mask = (y_test_ctrl>0.0001)
    ctrl.im_mask = t.tensor(im_mask)
    ctrl.nmeas = (im_mask==1).sum()
    ctrl.init_mhe(10,100,1)
    #errhere
    
    # compare analytical and acquired image with some phase applied
    #x_test = np.random.randn(ctrl_phimat.shape[0])
    x_test = np.r_[5.0,0.0,-0.5]
    y_test_ctrl = ctrl.h_func(t.tensor(x_test).to(t.float32)).detach().numpy()
    scale = 1.5
    slm.clear_phase()
    slm.add_piston_phase(0.5)
    slm.add_phimat_phase(u_ref)
    slm.add_phimat_phase(x_test*scale)
    slm.update()
    cv2.waitKey(100)
    with cam.FrameGrabber(cam) as fg:
        y_test_cam = (rebin(fg.grab(1)[0],cam_im_rebin)-cam_dark_frame)*flux_ratio
    plt.matshow(y_test_ctrl)
    plt.colorbar()
    plt.matshow(y_test_cam)
    plt.colorbar()
    plt.matshow(y_test_ctrl-y_test_cam)
    plt.colorbar()
    plt.title("difference with aberration")
    cv2.waitKey(1)
    
    #stophere
    #dark = 600
    sigma_tt = 1.0 # units are tbd
    sigma_foc = 2.0
    F_matrix = np.eye(3)
    sigma_x = np.concatenate([
            np.ones(2)*sigma_tt,
            np.ones(1)*sigma_foc
            ])
    Sigma_x = F_matrix @ np.diag(sigma_x) @ F_matrix.T
    a_matrix = 1.00*np.ones(3)
    A_matrix = np.diag(a_matrix) @ F_matrix
    Sigma_v = Sigma_x - A_matrix @ Sigma_x @ A_matrix.T
    w,v = np.linalg.eigh(Sigma_v)
    v = v[:,w>1e-9]
    w = w[w>1e-9]
    B_matrix = v @ np.diag(w**0.5)
    w,v = np.linalg.eigh(Sigma_x)
    v = v[:,w>1e-9]
    w = w[w>1e-9]
    S_matrix = v @ np.diag(w**0.5)

    disturbance = S_matrix @  np.random.randn(S_matrix.shape[1])
    # compare synth and real disturbance
    slm.clear_phase()
    slm.add_piston_phase(0.5)
    slm.add_phimat_phase(u_ref)
    slm.add_phimat_phase(disturbance)
    slm.update()
    cv2.waitKey(1000)
    with cam.FrameGrabber(cam) as fg:
        frame = (rebin(np.mean(fg.grab(100),axis=0)*1.0,cam_im_rebin)-cam_dark_frame)*flux_ratio
        plt.matshow(frame)
        plt.title("true image after disturbance")
        plt.colorbar()
        ctrl_frame = ctrl.h_func(t.tensor(disturbance).to(t.float32))
        plt.matshow(ctrl_frame)
        plt.title("synth image after disturbance")
        plt.colorbar()
    

    #disturbance = np.r_[10,-3,1,-1,1,-1,1,-1,1]*0.3
    #disturbance[2:] -= disturbance[2:].mean()
    fig,ax = plt.subplots(figsize=[4,4])
    ax.imshow(y_test_cam)
    # run control loop
    frames = []
    x_log = []
    u_log = []
    with cam.FrameGrabber(cam) as fg:
        for it in tqdm(range(500)):
            # update disturbance
            disturbance = a_matrix * disturbance + B_matrix @ np.random.randn(B_matrix.shape[1])
            # grab a frame
            frame = (rebin(fg.grab(1)[0]*1.0,cam_im_rebin)-cam_dark_frame)*flux_ratio
            ax.images[0].set_data(frame)
            frames.append(frame)
            slm.clear_phase()

            x_log.append(disturbance.copy())
            u_log.append(ctrl.x_corr*1.0)
            print(t.norm(t.tensor(x_log[-1])+u_log[-1],"fro"))
            
            if it > 20:
                # update command
                ctrl.update(frame[im_mask])
                ctrl.x_corr = t.einsum("ij,i->j",t.tensor(F_matrix).to(t.float32),ctrl.x_corr)
                #ctrl.x_corr[2] *= -1
                slm.add_phimat_phase(ctrl.x_corr*scale)

            # apply command
            slm.add_piston_phase(0.5)
            slm.add_phimat_phase(u_ref)
            slm.add_phimat_phase(disturbance*scale)
            slm.update()
            #print(ctrl.x_corr - disturbance)
            #print(((ctrl.x_corr - disturbance)**2).sum()**0.5)
            #print(frame.max())
            # wait a bit
            cv2.waitKey(100)

np.save("frames.npy",frames)
np.save("x.npy",np.array(x_log))
np.save("u.npy",np.array(u_log))