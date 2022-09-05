#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt; plt.ion()

from pripy.algos import GerchbergSaxton as GS
from tqdm import tqdm
from aotools import zernike

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
x_init = np.random.randn(10)*1
phase_s_0 = get_phase(x_init)

one_hot = np.zeros(x_init.shape)
one_hot[0] = 1.0
mode_to_phase = np.array([get_phase(np.roll(one_hot,i))[pup_s==1] for i in range(x_init.shape[0])]).T
phase_to_mode = np.linalg.solve(mode_to_phase.T @ mode_to_phase, mode_to_phase.T)

gs = GS(pup=pup_f,wavelength=2*np.pi,im_width=im_width,fft_width=fft_width ,offset=phase_offset)

gain = 0.3
leak = 0.99

# This cell uses just the constant x over nbuffer
niter = 50
x_dm = []
tar_phase = []
tar_im    = []
yd = []

x_corr = np.zeros(x_init.shape)

# uncorrected wavefront phase:
plt.matshow(phase_s_0+get_phase(x_corr))

y = trim_im(phase_to_image(phase_s_0+get_phase(x_corr)),trimmed_width=im_width).flatten()
ax = plt.matshow(y.reshape((im_width,im_width)))
plt.colorbar()

err = []
costs = []
x_a_old = x_corr*0
x_b_old = x_corr*0
dx = x_corr*0
x_corr_old = x_corr*0
for i in tqdm(range(niter),leave=False):
    y = trim_im(phase_to_image(phase_s_0+get_phase(x_corr)),trimmed_width=im_width)
    phi_a = gs.compute_phase(y,iters=20)
    phi_b = -phi_a[::-1,::-1]
    x_a = phase_to_mode @ phi_a[pup_f==1]
    x_b = phase_to_mode @ phi_b[pup_f==1]
    dx_hat = np.array([x_a - x_a_old,
                       x_a - x_b_old,
                       x_b - x_a_old,
                       x_b - x_b_old])
    which_min = np.argmin(((dx_hat - dx[None,:])**2).sum(axis=1))
    if which_min < 2:
        xk_opt = x_a
    else:
        xk_opt = x_b
    x_corr     = leak*x_corr - gain*xk_opt
    dx = x_corr - x_corr_old
    x_corr_old = x_corr.copy()
    x_a_old = x_a.copy()
    x_b_old = x_b.copy()
    ax.set_data(y.reshape((im_width,im_width)))
    ax.set_clim([y.min(),y.max()])
    err.append((phase_s_0+get_phase(x_corr))[pup_s==1].std())
    plt.title(f"Iteration {i:d}\nError: {err[-1]:0.3f}")
    plt.savefig("tame_%03d.png"%i)
    
# error over time
plt.figure()
plt.plot(err,label='error')
plt.title('Phase error')
plt.xlabel('Iteration')
plt.ylabel('Error [rad RMS]')
plt.legend()

# residual wavefront phase:
plt.matshow(phase_s_0+get_phase(x_corr))
plt.matshow(y.reshape([20,20]))