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
from pointgrey_handler import CameraHandler
import time
import matplotlib.pyplot as plt
from slm_make_gmt_pupil import make_phimat,make_pupil

plt.ion()

def rebin(a, factor):
    sh = a.shape[0]//factor,factor,a.shape[1]//factor,factor
    return a.reshape(sh).mean(-1).mean(1)

class Controller:
    u = np.r_[0.0,0.0]

    init = True            # True when requiring initialisation
    calibration_status = 0 # zero when finished or not yet started

    def __init__(self,im_width):
        self.im_width = im_width
        self._im_xx,self._im_yy = t.meshgrid(t.arange(im_width),t.arange(im_width))
        self._im_xx = self._im_xx.to(t.float32)
        self._im_yy = self._im_yy.to(t.float32)

    def step(self,frame): 
        # convert image8bit to mini display format
        cam_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # display image on window
        cv2.imshow('Cam img',cam_image)

        frame = frame.copy()
        frame[frame<=40] = 40
        frame -= 40
        frame = t.tensor(frame,dtype=t.float32)

        if self.init:
            self.calibrate(frame) 
            return
        
        cog = self.cog(frame)
        self.u = 0.999 * self.u - 0.1 * self.cog_gain @ cog
        print(self.u)
        self.set_command(self.u - self.cog_gain @ self.cog_ref)
        self.apply_command()
    
    def cog(self,frame):
        cog_x = frame.flatten() @ self._im_xx.flatten() / frame.flatten().sum()
        cog_y = frame.flatten() @ self._im_yy.flatten() / frame.flatten().sum()
        return np.r_[cog_x,cog_y]-self.im_width/2+0.5

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
    im_width = 200
    rebin_factor = 1
    ctrl = Controller(im_width=im_width)

    monitor_id = int(doc["--monitor"])
    radius = int(doc["--radius"])
    xoffset = int(doc["--xoffset"])
    yoffset = int(doc["--yoffset"])

    slm = SLM(monitor=monitor_id,isImageLock=True,wrap=True)

    full_width = 615 # full width of GMT pupil across long axis
    x0 = 455
    y0 = 285
    seg_diam = full_width/(3+0.359/8.4*2) # segment diameter in pixels
    
    phimat = make_phimat(x0=x0,y0=y0,x_res=slm._res_x,y_res=slm._res_y,
                        seg_diam=seg_diam)
    slm.set_phimat(phimat=phimat)

    cam  = CameraHandler(width=im_width,height=im_width,offset_x=420,offset_y=424)

    x = np.zeros(7,dtype=np.float32)
    with cam.FrameGrabber(cam) as fg:
        x *= 0.0
        slm.clear_phase()
        slm.add_phimat_phase(x)
        slm.update()
        cv2.waitKey(200)
        frame = np.mean(fg.grab(500),axis=0)
        frame = rebin(frame,rebin_factor)
        np.save("frame_norebin.npy",frame)
        # display image on window
        print(frame.max())
        plt.matshow(frame); plt.colorbar()
        x *= 0.0
        x += 0.5
        slm.clear_phase()
        slm.add_phimat_phase(x)
        slm.update()
        cv2.waitKey(200)
        frame = np.mean(fg.grab(500),axis=0)
        frame = rebin(frame,rebin_factor)
        print(frame.max())
        
        plt.matshow(frame); plt.colorbar()