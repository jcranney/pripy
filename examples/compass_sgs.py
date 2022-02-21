#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt; plt.ion()

from pripy.algos import GerchbergSaxton
from tqdm import tqdm

from shesha.supervisor.compassSupervisor import CompassSupervisor
from shesha.config import ParamConfig

config = ParamConfig("compass_par_tt.py")
sup = CompassSupervisor(config)
nwfs = 0
niter = 500

# simulation parameters
pup_width = sup.get_s_pupil().shape[0]
fft_width = sup.get_i_pupil().shape[0]
im_width = sup.wfs.get_wfs_image(nwfs).shape[0]
pup = sup.get_s_pupil()

# phase<->dm:
u1 = 0*sup.rtc.get_command(0)
sup.rtc.set_command(0,u1)
sup.rtc.apply_control(0,comp_voltage=False)
sup.wfs.raytrace(nwfs,dms=sup.dms)
sup.wfs.compute_wfs_image(nwfs)
h0 = sup.wfs.get_wfs_image(nwfs).flatten()
delta=1e2
u1[0] = delta
d_phi_dm = []
d_h_dm = []
for i in range(u1.shape[0]):
    sup.rtc.set_command(0,np.roll(u1,i))
    sup.rtc.apply_control(0,comp_voltage=False)
    sup.wfs.raytrace(nwfs,dms=sup.dms)
    d_phi_dm.append(sup.wfs.get_wfs_phase(nwfs)[sup.get_m_pupil()==1]/delta)
d_phi_dm = np.array(d_phi_dm).T
c_dm_phi = np.linalg.solve(d_phi_dm.T @ d_phi_dm + 1e-3*np.eye(d_phi_dm.shape[1]), d_phi_dm.T)

sup.reset()

# build BTT model:
w,v = np.linalg.eig(d_phi_dm.T @ d_phi_dm)
#v = v @ np.diag(w)
v = v[:,np.argsort(np.abs(w))[::-1]]

def get_phase_from_dm(u):
    sup.rtc.set_command(0,u)
    sup.rtc.apply_control(0,comp_voltage=False)
    sup.wfs.raytrace(nwfs,dms=sup.dms)
    return (sup.wfs.get_wfs_phase(nwfs)*sup.get_m_pupil())[2:-2,2:-2]

def get_phase_from_btt(x):
    return get_phase_from_dm(v[:,:x.shape[0]] @ x)

get_phase = lambda x : get_phase_from_btt(x)

nstate = 40
nmodes_max = 100
nmeas  = im_width**2

# phase to modes:
u = np.zeros(nmodes_max)
u[0] = 1.0
dphi = []
for i in range(nmodes_max):
    dphi.append(get_phase(np.roll(u,i))[pup==1])
dphi = np.array(dphi).T
cphi = np.linalg.solve(dphi.T @ dphi, dphi.T)[:nstate,:]
dphi = dphi[:,:nstate]

import cupy as cp
gs = GerchbergSaxton(pup=pup, wavelength=sup.config.p_wfss[nwfs].get_Lambda(),
                     fft_width=fft_width,im_width=im_width)

gain = 0.4
leak = 0.99

x_dm = []
tar_phase = []
tar_im    = []
yd = []

err = []
costs = []

x_corr = np.zeros(nstate)

sup.next()
ax = plt.matshow(sup.wfs.get_wfs_image(nwfs))

for i in range(niter):
    sup.next(apply_control=False)
    sup._print_strehl(1,1,1)
    y = sup.wfs.get_wfs_image(nwfs)
    sup.wfs.raytrace(nwfs,dms=sup.dms)
    x_dm.append(cphi @ (sup.wfs.get_wfs_phase(nwfs)[sup.get_m_pupil()==1]))
    
    phi0 = gs.compute_phase(y)
    phi1 = -phi0[::-1,::-1]

    xk_est0 = cphi @ phi0[pup==1]
    xk_est1 = cphi @ phi1[pup==1]

    errs = [(xk_est0-xk_est0_old)-(x_dm[-1]-x_dm[-2]),(xk_est0-xk_est1_old)-(x_dm[-1]-x_dm[-2]),
            (xk_est1-xk_est0_old)-(x_dm[-1]-x_dm[-2]),(xk_est1-xk_est1_old)-(x_dm[-1]-x_dm[-2])]
    which_est = np.argmin([np.std(e) for e in errs]) // 2
    if which_est == 0:
        xk_opt = xk_est0
    else:
        xk_opt = xk_est1

    x_corr = leak*x_corr - gain*((xk_opt))

    xk_est0_old = xk_est0.copy()
    xk_est1_old = xk_est1.copy()

    sup.rtc.set_command(0,v[:,:nstate] @ x_corr)
    sup.rtc.apply_control(0)
    ax.set_data(sup.wfs.get_wfs_image(nwfs))
    plt.savefig(f"eme_compass_{i:04d}.png")
    