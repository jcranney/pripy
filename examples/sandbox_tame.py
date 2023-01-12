#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from pripy.algos import MHEStatic; plt.ion()
#from pripy import TAME
from pripy import TaylorHModel
from pripy import MHE
from tqdm import tqdm
from aotools import zernike

import scipy.optimize as opt

# simulation parameters
pup_width = 200
fft_width = 512
im_width = 20

# build (circular) pupil
yy_s,xx_s = (np.mgrid[:pup_width,:pup_width]/(pup_width-1)-0.5)
pup_s = 1.0*((yy_s**2+xx_s**2)**0.5 <= 0.5)

pad_by = ((fft_width-pup_width)//2,(fft_width-pup_width)//2)
pup_f = np.pad(pup_s,pad_by)

# offset to get the image centred on 2x2 pixels
phase_offset = (-(xx_s+yy_s)*(pup_width-1)*2*np.pi/fft_width/2)

def phase_to_image(phi):
    """Take wavefront phase in small pupil dimensions and return image
    
    Args:
        phi (ndarray): phase in small pupil dimensions

    Returns:
        ndarray: focal plane image
    """
    wf_s = pup_s * np.exp(1j*(phi+phase_offset))
    wf_f = np.pad(wf_s,pad_by)
    im_F = np.abs(np.fft.fftshift(np.fft.fft2(wf_f))/pup_s.sum())**2
    return im_F

def trim_im(im,trimmed_width=im_width):
    """Trim image to a square of size trimmed_width x trimmed_width

    Args:
        im (ndarray): image to trim
        trimmed_width (int): size of the trimmed image

    Returns:
        ndarray: trimmed image
    """
    return im[
        im.shape[0]//2-trimmed_width//2:im.shape[0]//2+trimmed_width//2,
        im.shape[1]//2-trimmed_width//2:im.shape[1]//2+trimmed_width//2
        ]

def get_phase_zernike(x,pup_diam=pup_width,include_piston=False):
    """Get phase from modal coefficients
    
    Args:
        x (ndarray): modal coefficients
        pup_diam (int): size of the pupil

    Returns:
        ndarray: phase in pupil dimensions
    """
    if include_piston:
        phi = zernike.phaseFromZernikes(np.r_[x],pup_diam,norm="p2v")
    else:
        phi = zernike.phaseFromZernikes(np.r_[0.0,x],pup_diam,norm="p2v")
    return phi*pup_s

get_phase = get_phase_zernike
np.random.seed(74)
x_init = np.random.randn(10)*2
phase_s_0 = get_phase(x_init)

nstate = len(x_init)
nmeas  = im_width**2

h_taylor = TaylorHModel(3,nmeas,nstate)
h_taylor.general_build_dnys(pup_s,im_width,fft_width,get_phase,phase_offset,(np.pi*2))

nbuffer = 5
cost_scaling = 1.0
Sigma_x = 1*np.eye(nstate)
Sigma_w = 1e-5*np.eye(h_taylor.dny[0].shape[0])
A_mat = np.eye(nstate)*0.9999

#mve = MHE(nstate, nmeas, nbuffer, noise_cov=Sigma_w, state_cov=Sigma_x,
#            state_matrix=A_mat, h_eval=h_taylor.eval, h_jac=h_taylor.jacobian, 
#            h_hess=None, cost_scaling=cost_scaling)
mve = MHEStatic(nstate, nmeas, nbuffer, noise_cov=Sigma_w, state_cov=Sigma_x,
            h_eval=h_taylor.eval, h_jac=h_taylor.jacobian, 
            h_hess=None, cost_scaling=cost_scaling)

gain = 0.3
leak = 0.99

# This cell uses just the constant x over nbuffer
niter = 50
x_dm = []
tar_phase = []
tar_im    = []
yd = []

x_corr = np.zeros(nstate)

# uncorrected wavefront phase:
plt.matshow(phase_s_0+get_phase(x_corr))
plt.colorbar()
plt.title("Initial Wavefront")

# uncorrected image
y = trim_im(phase_to_image(phase_s_0+get_phase(x_corr)),trimmed_width=im_width).flatten()
plt.matshow(y.reshape((im_width,im_width)))
plt.colorbar()
plt.title("Initial Image")

err = []
costs = []
for i in tqdm(range(niter),leave=False):
    y = trim_im(phase_to_image(phase_s_0+get_phase(x_corr)),trimmed_width=im_width).flatten()
    yd.append(y/y.sum()*h_taylor.dny[0].sum())
    x_dm.append(x_corr)
    if i >= nbuffer:
        ydk   = np.array(yd[-nbuffer:])
        xk_dm = np.array(x_dm[-nbuffer:]).flatten()
        #x_0 = -np.tile(xk_dm[-nstate:],(nbuffer,1)).flatten()
        x_0 = -xk_dm[-nstate:]*0.0
        #errhere
        xk_opt = mve.get_estimate(x_0,xk_dm,ydk)
        x_corr    = (1-gain)*x_corr - gain*(xk_opt)
        costs.append(mve._xopt["fun"])
        #print(mve._xopt)
    err.append((phase_s_0+get_phase(x_corr))[pup_s==1].std())
    
# error over time
plt.figure()
plt.plot(err,label='error')
plt.title('Phase error')
plt.xlabel('Iteration')
plt.ylabel('Error [rad RMS]')
plt.legend()

# residual wavefront phase:
plt.matshow(phase_s_0+get_phase(x_corr))
plt.title("Final Wavefront")
plt.colorbar()

# residual image
plt.matshow(y.reshape([20,20]))
plt.colorbar()
plt.title("Final Image")