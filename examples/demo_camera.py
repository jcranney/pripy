# -*- coding: utf-8 -*-
"""
Usage: test_slm.py [options]

    -s, --slm               Use the SLM
    -m, --monitor <id>      The monitor to use, -1 means virtual SLM [default: -1]
    -r, --radius <pixels>   Radius of pupil to use in pixels [default: 300]
    -x, --xoffset <pixels>  x offset in pixels [default: 0]
    -y, --yoffset <pixels>  y offset in pixels [default: 0]
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
from slm_make_gmt_pupil import make_phimat,make_pupil,make_tt_phimat
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

        self.gain = 0.4
        self.leak = 0.995

        self.x_dm = t.zeros([self.nbuffer,self.nstate])
        self.yd = t.zeros([self.nbuffer,self.nmeas])
        self.x_corr = t.zeros(self.nstate)

        self.iter = 0

    def cost_ls(self,x,u,y):
        z = t.zeros(self.nbuffer*self.nmeas)
        for n in range(self.nbuffer):
            tmp = y[n]-self.h_func(x+u[n]).flatten()
            z[n*self.nmeas:(n+1)*self.nmeas] = t.einsum("i,ij->j",tmp,self.Sigma_w_inv_fact)
        #return t.concat([z,1e-4*x],dim=0)
        return z

    def update(self,frame):
        self.yd   = self.yd.roll(-1,0)
        self.x_dm = self.x_dm.roll(-1,0)
        self.yd[-1,:]   = t.tensor(frame.flatten()).to(t.float32)
        self.x_dm[-1,:] = self.x_corr

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

    # TODO Delet ths
    def calibrate(self,frame):
        if self.calibration_status == 0:
            self.calibration_status = 1 # getting reference slope
            self.set_command(np.r_[0.0,0.0])
            self.t_settled = time.time() + 2.0
        elif self.calibration_status == 1:
            if time.time() > self.t_settled:
                self.calibration_status = 2 # reference slope frame available
                self.cog_ref = self.cog(frame)
                self.set_command(np.r_[1,0.0]) # get tip response
                self.t_settled = time.time() + 2.0
        elif self.calibration_status == 2:
            if time.time() > self.t_settled:
                self.calibration_status = 3 # reference slope frame available
                self.cog_ref_tip = self.cog(frame)-self.cog_ref
                self.set_command(np.r_[0.0,1.0]) # get tilt response
                self.t_settled = time.time() + 2.0
        elif self.calibration_status == 3:
            if time.time() > self.t_settled:
                self.calibration_status = 0 # reference slope frame available
                self.cog_ref_tilt = self.cog(frame)-self.cog_ref
                self.set_command(np.r_[0.0,0.0])
                self.init = False
                print(self.cog_ref_tip)
                print(self.cog_ref_tilt)
                self.cog_gain = np.linalg.inv(np.array([self.cog_ref_tip,self.cog_ref_tilt]))
        self.apply_command()

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

    # slm params
    slm_monitor_id = int(doc["--monitor"])
    slm_full_width = 615 # full width of GMT pupil across long axis
    slm_x0 = 455
    slm_y0 = 285
    slm_seg_diam = slm_full_width/(3+0.359/8.4*2) # segment diameter in pixels
    

    # camera params
    cam_im_width_full = 56
    cam_im_rebin = 4
    assert(cam_im_width_full % cam_im_rebin == 0)
    cam_im_width = cam_im_width_full // cam_im_rebin
    cam_offset_x = 464
    cam_offset_y = 520
    cam_pixel_as = 1.0 # TODO compute this from physical system parameters
    cam_dark_frame = np.load("dark.npy")

    # initialise slm
    slm = SLM(monitor=slm_monitor_id,isImageLock=True,wrap=True)
    slm_phimat = np.concatenate([
        make_tt_phimat(x0=slm_x0, y0=slm_y0, 
                x_res=slm._res_x, y_res=slm._res_y,
                seg_diam=slm_seg_diam),
        make_phimat(x0=slm_x0, y0=slm_y0, 
                x_res=slm._res_x, y_res=slm._res_y,
                seg_diam=slm_seg_diam)],axis=0)
    slm.set_phimat(phimat=slm_phimat*0.14)
    slm.clear_phase()
    slm.update()

    # initialise camera
    if 1==1:
        cam  = CameraHandler(width=cam_im_width_full, height=cam_im_width_full,
                    offset_x=cam_offset_x, offset_y=cam_offset_y,
                    exposure=100e3, gain=5.0)

        with cam.FrameGrabber(cam) as fg:
            fig,ax = plt.subplots(figsize=[4,4])
            ax.imshow(fg.grab(1)[0])
            while True:
                ax.images[0].set_data(fg.grab(1)[0])
                cv2.waitKey(10)
    else:
        while True:
            cv2.waitKey(10)