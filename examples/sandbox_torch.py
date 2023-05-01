#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np_lazy

plt.ion()
from tqdm import tqdm
from aotools import zernike
import torch as t
t.set_num_threads(6)

# simulation parameters
pup_width = 200
fft_width = 512
im_width = 8
MAX_ZERN = 50

# build (circular) pupil
yy_s,xx_s = (np_lazy.mgrid[:pup_width,:pup_width]/(pup_width-1)-0.5)
pup_s = 1.0*((yy_s**2+xx_s**2)**0.5 <= 0.5)
pup_s = t.tensor(pup_s,dtype=t.float32)

# offset to get the image centred on 2x2 pixels
phase_offset = t.tensor(-(xx_s+yy_s)*(pup_width-1)*2*t.pi/fft_width/2,dtype=t.float32)

def phase_to_image(phi):
    """Take wavefront phase in small pupil dimensions and return image
    
    Args:
        phi (ndarray): phase in small pupil dimensions

    Returns:
        ndarray: focal plane image
    """
    wf_s = pup_s * t.exp(1j*(phi+phase_offset))
    im_F = t.abs(t.fft.fftshift(t.fft.fft2(wf_s,s=[fft_width,fft_width]))/pup_s.sum())**2
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

zern_array = t.tensor(zernike.zernikeArray(MAX_ZERN,pup_width,norm="p2v"),dtype=t.float32)

def get_phase_zernike(x,pup_diam=pup_width,include_piston=False):
    """Get phase from modal coefficients
    
    Args:
        x (ndarray): modal coefficients
        pup_diam (int): size of the pupil

    Returns:
        ndarray: phase in pupil dimensions
    """
    if include_piston:
        phi = t.einsum("ijk,i->jk",zern_array[:len(x),:,:],x)
    else:
        phi = t.einsum("ijk,i->jk",zern_array[1:len(x)+1,:,:],x)
    return phi*pup_s

get_phase = get_phase_zernike
t.random.manual_seed(74)
x_init = t.randn(10)*1.0
phase_s_0 = get_phase(x_init)

nstate = len(x_init)
nmeas  = im_width**2

def h_func(x):
    "takes state, returns image"
    return trim_im(phase_to_image(get_phase(x)),trimmed_width=im_width)

nbuffer = 10
cost_scaling = 1.0
#A_mat = t.eye(nstate)*0.9999
Sigma_x = 1000.0*t.eye(nstate)
#Sigma_v = Sigma_x - A_mat @ Sigma_x @ A_mat.T
#Sigma_v_inv = t.linalg.inv(Sigma_v)
Sigma_x_inv = t.linalg.inv(Sigma_x)
Sigma_w_inv = 1.0*t.eye(nmeas)

#Sigma_v_inv_fact = t.linalg.cholesky(Sigma_v_inv)
Sigma_x_inv_fact = 0*t.linalg.cholesky(Sigma_x_inv)
Sigma_w_inv_fact = t.linalg.cholesky(Sigma_w_inv)

"""
def cost(x,u,y):
    cost = t.tensor(0.0)
    for n in range(nbuffer-1):
        tmp = x[n+1]-t.einsum("ij,j->i",A_mat,x[n])
        cost += 0.5*t.einsum("i,ij,j->",tmp,Sigma_v_inv,tmp)
    for n in range(nbuffer):
        tmp = y[n]-h_func(x[n]+u[n]).flatten()
        cost += 0.5*t.einsum("i,ij,j->",tmp,Sigma_w_inv,tmp)
    cost += 0.5*t.einsum("i,ij,j->",x[0],Sigma_x_inv,x[0])
    return cost

x = t.randn([nbuffer,nstate])/100
x.requires_grad_(True)
optimizer = t.optim.SGD([x],lr=0.0001)

def cost_one(x,u,y):
    cost = t.tensor(0.0)
    for n in range(nbuffer):
        tmp = y[n]-h_func(x+u[n]).flatten()
        cost += 0.5*t.einsum("i,ij,j->",tmp,Sigma_w_inv,tmp)
    cost += 0.5*t.einsum("i,ij,j->",x,Sigma_x_inv,x)
    return cost

"""

def cost_ls(x,u,y):
    z = t.zeros(nbuffer*nmeas)#+nstate)
    #z[:nstate] = 0.5**0.5*t.einsum("i,ij->j",x,Sigma_x_inv_fact)
    for n in range(nbuffer):
        tmp = y[n]-h_func(x+u[n]).flatten()
        z[n*nmeas:(n+1)*nmeas] = 0.5**0.5*t.einsum("i,ij->j",tmp,Sigma_w_inv_fact)
    return z

x = t.randn([nstate])*0
x.requires_grad_(True)
optimizer = t.optim.SGD([x],lr=0.01)

gain = 0.5
leak = 1.0

# This cell uses just the constant x over nbuffer
niter = 100
x_dm = t.zeros([nbuffer,nstate])
yd = t.zeros([nbuffer,nmeas])
x_corr = t.zeros(nstate)

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
print(x_init)
flux = 1000.0
for i in tqdm(range(niter),leave=False):
    y = t.poisson(trim_im(phase_to_image(phase_s_0+get_phase(x_corr)),trimmed_width=im_width).flatten()*flux)/flux
    yd   = yd.roll(-1,0)
    x_dm = x_dm.roll(-1,0)
    yd[-1,:]   = y
    x_dm[-1,:] = x_corr
    if i >= nbuffer:
        #errhere
        #with t.no_grad():
        #    x *= 0.0
        mu = 0.001
        nu = 2.0
        ep_1 = ep_2 = 1e-6
        jac = t.autograd.functional.jacobian(lambda x : cost_ls(x,x_dm,yd),x,vectorize=True,strategy="forward-mode")
        f_x = cost_ls(x,x_dm,yd)
        old_cost = (f_x**2).sum()
        for it in tqdm(range(20),leave=False):
            dx  = -t.linalg.solve(jac.T @ jac + mu * t.eye(nstate), jac.T @ f_x)
            if t.norm(dx,"fro") <= ep_2*(t.norm(x,"fro")+ep_2):
                break
            x_new = x + dx
            cost = (cost_ls(x_new,x_dm,yd)**2).sum()
            rho = (old_cost - cost)/(0.5*dx @ (mu*dx - jac.T @ f_x))
            if rho > 0:
                with t.no_grad():
                    x += dx
                jac = t.autograd.functional.jacobian(lambda x : cost_ls(x,x_dm,yd),x,vectorize=True,strategy="forward-mode")
                f_x = cost_ls(x,x_dm,yd)
                old_cost = (f_x**2).sum()
                mu = mu * max(1/3,1-(2*rho-1)**3)
                nu   = 2
            else:
                mu = mu * nu
                nu = 2 * nu
        print(old_cost)
        #xk_opt = x.clone().detach().numpy()[-1,:]
        xk_opt = x.clone().detach().numpy()
        x_corr    = leak*(1-gain)*x_corr - gain*(xk_opt)
        #print(loss)
        print(x-x_init)
    err.append((phase_s_0+get_phase(x_corr))[pup_s==1].std())
print(err[-1])

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
plt.matshow(y.reshape([im_width,im_width]))
plt.colorbar()
plt.title("Final Image")
