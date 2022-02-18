#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt; plt.ion()

from pripy.algos import MHE, MHEStatic
from pripy import TaylorHModel
from tqdm import tqdm
from aotools import zernike

from shesha.supervisor.compassSupervisor import CompassSupervisor
from shesha.config import ParamConfig

config = ParamConfig("compass_ngs.py")
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
    sup.wfs.compute_wfs_image(nwfs,noise=False)
    d_phi_dm.append(sup.wfs.get_wfs_phase(nwfs)[sup.get_m_pupil()==1]/delta)
    d_h_dm.append(sup.wfs.get_wfs_image(nwfs).flatten()-h0)
d_phi_dm = np.array(d_phi_dm).T
c_dm_phi = np.linalg.solve(d_phi_dm.T @ d_phi_dm + 1e-3*np.eye(d_phi_dm.shape[1]), d_phi_dm.T)
d_h_dm = np.array(d_h_dm).T

ww,vv = np.linalg.eig(d_h_dm.T @ d_h_dm)
vv = vv[:,np.argsort(np.abs(ww))[::-1]]

sup.reset()

# Would like a modal basis that is determined by WFS sensitivity to that mode. 
# Maybe the diagonal of the hessian solved at the origin would be a good indicator of sensitivity?

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

def get_phase_from_weird(x):
    return get_phase_from_dm(vv[:,:x.shape[0]] @ x)

get_phase = lambda x : get_phase_from_btt(x)
#get_phase = lambda x : get_phase_from_weird(x)
#v = vv

nstate = 20
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

h_taylor = TaylorHModel(3,nmeas,nstate)
h_taylor.compass_build_dnys(sup,nwfs,get_phase)

nbuffer = 10
cost_scaling = 1e-9
Sigma_x = 1e6*0.05*np.eye(nstate)
Sigma_w = 1e6*1e-5*np.eye(h_taylor.dny[0].shape[0])
A_mat = np.eye(nstate)*0.9999

#mve = MHE(nstate, nmeas, nbuffer, noise_cov=Sigma_w, state_cov=Sigma_x,
#            state_matrix=A_mat, h_eval=h_taylor.eval, h_jac=h_taylor.jacobian, 
#            h_hess=None, cost_scaling=cost_scaling)
#mve = MHEStatic(nstate, nmeas, nbuffer, noise_cov=Sigma_w, state_cov=Sigma_x,
#            state_matrix=A_mat, h_eval=h_taylor.eval, h_jac=h_taylor.jacobian, 
#            h_hess=None, cost_scaling=cost_scaling)
import cupy as cp
mve = MHEStatic(nstate, nmeas, nbuffer, noise_cov=Sigma_w, state_cov=Sigma_x,
            state_matrix=A_mat, h_eval=lambda x: h_taylor.h_true_cp(cp.array(x)).get(),
            h_jac=h_taylor.exact_jacobian, h_hess=None, cost_scaling=cost_scaling)

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
    y = sup.wfs.get_wfs_image(nwfs).flatten()
    yd.append(y/y.sum()*h_taylor.dny[0].get().sum())
    resi_x = cphi @ sup.wfs.get_wfs_phase(nwfs)[sup.get_m_pupil()==1]
    sup.wfs.raytrace(nwfs,atm=sup.atmos,tel=sup.tel)
    true_x = cphi @ sup.wfs.get_wfs_phase(nwfs)[sup.get_m_pupil()==1]
    print("true resi: " + "".join([f"{x:7.1f}," for x in resi_x]))
    print("true turb: " + "".join([f"{x:7.1f}," for x in true_x]))
    sup.wfs.raytrace(nwfs,dms=sup.dms)
    x_dm.append((cphi @ sup.wfs.get_wfs_phase(nwfs)[sup.get_m_pupil()==1]))
    print(" dm shape: " + "".join([f"{x:7.1f}," for x in x_dm[-1]]))
    if i >= nbuffer:
        ydk   = np.array(yd[-nbuffer:])
        xk_dm = np.array(x_dm[-nbuffer:]).flatten()
        x_0 = -xk_dm[-nstate:]
        xk_opt = mve.get_estimate(x_0,xk_dm,ydk)
        print(" est turb: "+"".join([f"{x:7.1f}," for x in xk_opt]))
        print(" est resi: "+"".join([f"{x:7.1f}," for x in (xk_opt+xk_dm[-nstate:])]))
        x_corr    = leak*x_corr - gain*((xk_opt+xk_dm[-nstate:]))
        costs.append(mve._xopt["fun"])
    sup.rtc.set_command(0,v[:,:nstate] @ x_corr)
    sup.rtc.apply_control(0)
    ax.set_data(sup.wfs.get_wfs_image(nwfs))
    plt.savefig(f"eme_compass_{i:04d}.png")
    