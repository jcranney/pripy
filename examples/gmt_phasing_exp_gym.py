# -*- coding: utf-8 -*-
"""
Usage: test_slm.py [options]

    -s, --slm               Use the SLM
    -m, --monitor <id>      The monitor to use, -1 means virtual SLM [default: -1]
"""

from docopt import docopt
import numpy as np
import torch as t
t.set_num_threads(6)
import time
import matplotlib.pyplot as plt
from slm_make_gmt_pupil import make_phimat,make_pupil,make_tt_phimat,make_foc_phimat
from tqdm import tqdm
from scipy.linalg import block_diag

import gymnasium as gym
import gym_gmtphasing
import numpy as np

env = gym.make("gmtphasing-v0")
observation, info = env.reset(seed=42, return_info=True)

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

    def trim_im(self,im,do_rebin=True):
        """Trim image to a square of size trimmed_width x trimmed_width

        Args:
            im (ndarray): image to trim
            trimmed_width (int): size of the trimmed image

        Returns:
            ndarray: trimmed image
        """
        if do_rebin:
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

if __name__ == "__main__":
    doc = docopt(__doc__)

    # parameters (eventually should be moved to cli/docopt)
    # global params
    wavelength = 0.633

    # controller params
    ctrl_im_width_full = 28
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
        make_phimat(**ctrl_pup_kwargs)],axis=0
        )).to(t.float32)
    ctrl_fft_width = 384

    # initialise controller
    ctrl = Controller(im_width=ctrl_im_width, pup_width=ctrl_pup_width,
                fft_width=ctrl_fft_width, rebin_factor=ctrl_im_rebin,
                nstate=ctrl_phimat.shape[0], pup_s=ctrl_pup_s,
                get_phase = lambda x : t.einsum("ijk,i->jk",ctrl_phimat,x))

    # compare analytical and acquired image with no phase applied
    scale = 3.5
    x_test_ctrl = np.zeros(ctrl_phimat.shape[0],dtype=np.float32)
    y_test_ctrl = ctrl.h_func(t.tensor(x_test_ctrl).to(t.float32)).detach().numpy()
    frame,_,_,_ = env.step(np.float32(x_test_ctrl*scale))
    frame = ctrl.trim_im(frame,False)
    y_test_cam = frame
    flux_ratio = y_test_ctrl.sum()/y_test_cam.sum()
    diff = y_test_ctrl - y_test_cam*flux_ratio
    fig,ax = plt.subplots()
    ax.imshow(diff)
    plt.pause(1e-3)

    # compare analytical and acquired image with tip applied
    x_test_ctrl = np.zeros(ctrl_phimat.shape[0],dtype=np.float32)
    x_test_ctrl[0] = 5.0
    y_test_ctrl = ctrl.h_func(t.tensor(x_test_ctrl).to(t.float32)).detach().numpy()
    frame,_,_,_ = env.step(np.float32(x_test_ctrl*scale))
    frame = ctrl.trim_im(frame,False)
    env.step(np.float32(-x_test_ctrl*scale))
    y_test_cam = frame*flux_ratio
    diff = y_test_ctrl - y_test_cam
    ax.images[0].set_data(diff)
    ax.images[0].set_clim([diff.min(),diff.max()])
    plt.pause(1e-3)

    im_mask = (y_test_ctrl>0.00)
    ctrl.im_mask = t.tensor(im_mask)
    ctrl.nmeas = (im_mask==1).sum()
    ctrl.init_mhe(20,100,1)
    
    sigma_piston = 0.0
    sigma_tt = 0.0
    F_matrix = block_diag(np.eye(2),np.eye(7)-1/7*np.ones([7,7]))
    sigma_x = np.concatenate([
        np.ones(2)*sigma_tt,
        np.ones(7)*sigma_piston],
        axis=0)
    Sigma_x = F_matrix @ np.diag(sigma_x) @ F_matrix.T
    a_matrix = 0.99*np.ones(sigma_x.shape[0])
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

    disturbance = (S_matrix @  np.random.randn(S_matrix.shape[1])).astype(np.float32)

    # run control loop
    frames = []
    x_log = []
    u_log = []
    x_corr = np.zeros(env.env.nactu,dtype=np.float32)
    x_corr_old = np.zeros(env.env.nactu,dtype=np.float32)
    for it in tqdm(range(500)):
        # update disturbance
        disturbance[:] = a_matrix * disturbance + B_matrix @ np.random.randn(B_matrix.shape[1])
        # grab a frame
        frame,_,_,_ = env.step(((x_corr-x_corr_old)*scale).astype(np.float32))

        frame = frame*flux_ratio
        flux = 3000.0
        frame = (np.random.poisson(frame*flux)+10*np.random.randn(*frame.shape))/flux
        frame = ctrl.trim_im(frame,do_rebin=False)
        ax.images[0].set_data(frame)
        ax.images[0].set_clim([frame.min(),frame.max()])
        ax.set_title(f"{frame.max():0.3f}")
        plt.pause(1e-2)
        frames.append(frame.copy())

        x_log.append(disturbance.copy())
        u_log.append(ctrl.x_corr*1.0)
        
        if it > 20:
            # update command
            ctrl.update(frame[im_mask])
            x_corr_old[:] = x_corr[:]
            x_corr[:] = t.einsum("ij,i->j",t.tensor(F_matrix).to(t.float32),ctrl.x_corr)

    np.save("frames_gym.npy",frames)
    np.save("x_gym.npy",np.array(x_log))
    np.save("u_gym.npy",np.array(u_log))