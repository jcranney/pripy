#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
#from pripy import TAME
from pripy import TaylorHModel
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

# constant wavefront phase to be corrected
phase_s_0 = 0.2*np.sin(xx_s*2*np.pi*3)+0.5*np.cos((xx_s+yy_s)*2*np.pi*2)

def get_phase_perfect(x,pup_diam=pup_width,piston=0.0):
    """Get phase from modal coefficients, but only using the two modes
    present in the aberration.
    
    Args:
        x (ndarray): modal coefficients
        pup_diam (int): size of the pupil

    Returns:
        ndarray: phase in pupil dimensions
    """
    phi = x[0]*np.sin(xx_s*2*np.pi*3)+x[1]*np.cos((xx_s+yy_s)*2*np.pi*2)
    #phi = zernike.phaseFromZernikes(np.r_[piston,x],pup_diam,norm="p2v")
    return phi*pup_s

def get_phase(x,pup_diam=pup_width,piston=0.0):
    """Get phase from modal coefficients
    
    Args:
        x (ndarray): modal coefficients
        pup_diam (int): size of the pupil

    Returns:
        ndarray: phase in pupil dimensions
    """
    phi = zernike.phaseFromZernikes(np.r_[piston,x],pup_diam,norm="p2v")
    return phi*pup_s

nstate=30

nstate=2
get_phase = get_phase_perfect

h_taylor = TaylorHModel(3,im_width**2,nstate)
h_taylor.general_build_dnys(pup_s,im_width,fft_width,get_phase,phase_offset,(np.pi*2))

x0 = np.zeros(nstate)

# big assumption that they are all decoupled:
SigX_mech = (0.1)**2*np.ones(nstate)
SigXp1_mech = 0.99*SigX_mech
mask_mech = np.zeros(nstate)
mask_mech[1:1+7] = 1.0
A_mech = (SigXp1_mech/SigX_mech)*mask_mech
B_mech = mask_mech*(SigX_mech-A_mech**2*SigX_mech)**0.5
x_mech = mask_mech*(np.random.randn(nstate)*(SigX_mech**0.5))

N_BUFFER = 5
scaling_factor = 1e-5

def cost(x,x_dm,yd,h_eval,Gamma,Sigma_w_inv):
    """
    x should be (N,NSTATE).
    """
    J  = ((x).T @ Gamma @ (x))
    h  = np.r_[[h_eval(x+(x_dm)[nstate*i:nstate*(i+1)]) for i in range(N_BUFFER)]]
    J += np.sum([hi.T @ Sigma_w_inv @ hi for hi in h],axis=0)
    J += - 2 * np.sum([hi.T @ Sigma_w_inv @ ydi for hi,ydi in zip(h,yd)],axis=0)
    return J*scaling_factor

def jac(x,x_dm,yd,h_eval,h_jac,Gamma,Sigma_w_inv):
    """
    x should be (N,NSTATE)
    """
    h     = [h_eval(x+(x_dm)[nstate*i:nstate*(i+1)]) for i in range(N_BUFFER)]
    dhdx  = [ h_jac(x+(x_dm)[nstate*i:nstate*(i+1)]) for i in range(N_BUFFER)]
    djdx  = 2*(Gamma @ x)
    djdx += 2*np.sum([dhidx.T @ Sigma_w_inv @ hi for dhidx,hi in zip(dhdx,h)],axis=0)
    djdx -= 2*np.sum([dhidx.T @ Sigma_w_inv @ ydi for dhidx,ydi in zip(dhdx,yd)],axis=0)
    return djdx*scaling_factor

def hess(x,x_dm,yd,h_eval,h_jac,h_hess,Gamma,Sigma_w_inv):
    """
    x should be (N,NSTATE)
    """
    h       = [h_eval(x+(x_dm)[nstate*i:nstate*(i+1)]) for i in range(N_BUFFER)]
    dhdx    = [ h_jac(x+(x_dm)[nstate*i:nstate*(i+1)]) for i in range(N_BUFFER)]
    d2hdx2  = [h_hess(x+(x_dm)[nstate*i:nstate*(i+1)]) for i in range(N_BUFFER)]
    d2jdx2  = 2*(Gamma)
    d2jdx2 += 2*np.sum([d2hidx2.T @ Sigma_w_inv @ hi for d2hidx2,hi in zip(d2hdx2,h)],axis=0)
    d2jdx2 += 2*np.sum([dhidx.T @ Sigma_w_inv @ dhidx for dhidx in dhdx],axis=0)
    d2jdx2 -= 2*np.sum([d2hidx2.T @ Sigma_w_inv @ ydi for d2hidx2,ydi in zip(d2hdx2,yd)],axis=0)
    return d2jdx2*scaling_factor

h_eval = h_taylor.eval
h_jac  = h_taylor.jacobian
h_hess = h_taylor.hessian

Sigma_x = 0.01*np.eye(nstate)
Sigma_x_inv = np.linalg.inv(Sigma_x)
Gamma = Sigma_x_inv
Sigma_w_inv = (1)*np.eye(h_taylor.dny[0].shape[0])

nactu = nstate

gain = 0.5
leak = 0.99

# This cell uses just the constant x over nbuffer
niter = 50
x_dm = []
tar_phase = []
tar_im    = []
xk_opt = None
yd = []
y_phase = []
x_dm_save = []

x_corr = np.zeros(nstate)
x_corr = np.random.randn(2)

# uncorrected wavefront phase:
plt.matshow(phase_s_0-get_phase(x_corr))
err = []
costs = []
for i in range(niter):
    y = trim_im(phase_to_image(phase_s_0-get_phase(x_corr)),trimmed_width=im_width).flatten()
    yd.append(y/y.sum()*h_taylor.dny[0].get().sum())
    x_dm.append(-x_corr)
    if i >= 10:
        ydk   = np.array(yd[-N_BUFFER:])
        xk_dm = np.array(x_dm[-N_BUFFER:]).flatten()
        x_0 = -xk_dm[-nstate:]
        xk_opt = opt.minimize(lambda x: cost(x,xk_dm,ydk,h_eval,Gamma,Sigma_w_inv),
                             x0,
                             jac=lambda x: jac(x,xk_dm,ydk,h_eval,h_jac,Gamma,Sigma_w_inv),
                             #hess=lambda x: hess(x,xk_dm,ydk,h_eval,h_jac,h_hess,Gamma,Sigma_w_inv),
                             #method="Newton-CG",options={'xtol': 1e-09,"maxiter":100})
                             method="BFGS")
        print((xk_opt["x"]+xk_dm[-nstate:]))
        x_corr    = leak*x_corr + gain*((xk_opt["x"]+xk_dm[-nstate:]))
        print(xk_opt)
    err.append((phase_s_0-get_phase(x_corr))[pup_s==1].std())
    costs.append([xk_opt["fun"] if xk_opt is not None else np.nan])
    print(f"{i:4d} : {err[-1]:0.3f} : [{x_corr[0]:9.5f} , {x_corr[1]:9.5f} ] : {[xk_opt['status'] if xk_opt else -1][0]:d}")

# error over time
plt.figure()
plt.subplot(311)
plt.plot(err,label='error')
plt.title('Phase error')
plt.xlabel('Iteration')
plt.ylabel('Error [rad RMS]')
plt.legend()

plt.subplot(312)
plt.plot(np.r_[[yd[i].max()/h_taylor.dny[0].get().max() for i in range(i+1)]],label="strehl")

plt.subplot(313)
plt.plot(np.r_[costs].flatten(),label='cost')

# residual wavefront phase:
plt.matshow(phase_s_0-get_phase(x_corr))

