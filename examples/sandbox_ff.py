#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt; plt.ion()

from pripy import FastAndFurious
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
x_init = np.random.randn(10)*2
phase_s_0 = get_phase(x_init)

nmodes = len(x_init)
nmeas  = im_width**2

# build phase projection inversion matrix
x1 = np.zeros(nmodes+1)
x1[0] = 1.0
dphi = np.array([get_phase_zernike(np.roll(x1,i),include_piston=True)[pup_s==1]
                        for i in range(len(x1))]).T
phase_to_modes = np.linalg.solve(dphi.T @ dphi, dphi.T)[1:,:]

# regularisation parameters
epsilon = np.pad(np.ones([im_width,im_width])*1e-5,(fft_width-im_width)//2,mode='constant',constant_values=1e5)

# create Fast and Furious instance
ff = FastAndFurious(pup=pup_s,fft_width=pup_f.shape[0],
                    im_width=im_width,offset=phase_offset,epsilon=epsilon)

gain = 0.5
leak = 0.99

# This cell uses just the constant x over nbuffer
niter = 50

# correcting wavefront phase:
phase_corr     = phase_s_0 * 0.0
phase_corr_old = phase_s_0 * 0.0

x_corr = np.zeros(nmodes)

# uncorrected wavefront phase:
plt.matshow(phase_s_0+get_phase(x_corr))
plt.title("Uncorrected wavefront phase")
plt.colorbar()


y = trim_im(phase_to_image(phase_s_0+get_phase(x_corr)),trimmed_width=im_width)
plt.matshow(y)
plt.title("Initial WFS image")
plt.colorbar()
ax = plt.matshow(y)
plt.colorbar()

err = []
costs = []
for i in tqdm(range(niter),leave=False):
    #### FF code, to be made compatible with the TAME code,
    #### Also, would like to do an intermediate modal projection, rather than
    #### controlling the phase directly.
    y = trim_im(phase_to_image(phase_s_0+phase_corr),trimmed_width=im_width)
    phi_ff = ff.compute_phase(y)
    xk_opt = phase_to_modes @ phi_ff[pup_s==1]
    x_corr *= leak
    x_corr += -gain*xk_opt
    phase_corr_old = phase_corr.copy()
    phase_corr = get_phase(x_corr)
    ff.set_diversity_phase((phase_corr-phase_corr_old))

    ax.set_data(y)
    ax.set_clim([y.min(),y.max()])
    err.append((phase_s_0+phase_corr)[pup_s==1].std())
    plt.title(f"Iteration {i:d}\nError: {err[-1]:0.3f}")
    plt.savefig("ff_%03d.png"%i)
    
# error over time
plt.figure()
plt.plot(err,label='error')
plt.title('Phase error')
plt.xlabel('Iteration')
plt.ylabel('Error [rad RMS]')
plt.legend()

# residual wavefront phase:
plt.matshow(phase_s_0+get_phase(x_corr))
plt.title("Residual wavefront phase")
plt.colorbar()