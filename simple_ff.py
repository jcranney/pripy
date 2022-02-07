import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from pripy.util import FastAndFurious

pup_diam = 500
fourier_diam = 1024
im_diam = 20
yy_s,xx_s = (np.mgrid[:pup_diam,:pup_diam]/(pup_diam-1)-0.5)
pup_s = 1.0*((yy_s**2+xx_s**2)**0.5 <= 0.5)
#pup_s = np.load('/home/jcranney/gmt-oiwfs/compass/gmt_pup_500.npy')
pad_list = [(fourier_diam-pup_diam)//2,(fourier_diam-pup_diam)//2]
pup_f = np.pad(pup_s,pad_list)
phase_offset = (-(xx_s+yy_s)*(pup_diam-1)*2*np.pi/fourier_diam/2)

def phase_to_image(phi):
    wf_s = pup_s * np.exp(1j*(phi+phase_offset))
    wf_f = np.pad(wf_s,pad_list)
    im_F = np.abs(np.fft.fftshift(np.fft.fft2(wf_f))/pup_s.sum())**2
    return im_F

def trim_im(im,trimmed_width=im_diam):
    return im[
        im.shape[0]//2-trimmed_width//2:im.shape[0]//2+trimmed_width//2,
        im.shape[1]//2-trimmed_width//2:im.shape[1]//2+trimmed_width//2
        ]

epsilon = np.pad(np.ones([im_diam,im_diam])*1e-5,(fourier_diam-im_diam)//2,mode='constant',constant_values=1e5)

ff = FastAndFurious(pup=pup_s,fft_width=pup_f.shape[0],
                    im_width=im_diam,offset=phase_offset,epsilon=epsilon)

phase_s_0      = 0.5*np.sin(xx_s*2*np.pi*3)+0.2*np.cos((xx_s+yy_s)*2*np.pi*2)
phase_corr     = phase_s_0 * 0.0
phase_corr_old = phase_s_0 * 0.0
im_S = trim_im(phase_to_image(phase_s_0+phase_corr),trimmed_width=im_diam)

#p_ie,S,a2,y_i = ff.get_phase(im_F)
#errorhere
phi_ff = pup_s.copy()
phi_ff = ff.retrieve_phase(im_S)
gain = 0.05
phase_diversity = -gain*phi_ff.copy()

plt.matshow(phi_ff)
plt.matshow(phase_s_0+phase_corr)
err = []
for i in range(10000):
    phase_corr *= 0.99
    phase_corr += phase_diversity
    ff.set_diversity_phase((phase_corr-phase_corr_old))
    im_S = trim_im(phase_to_image(phase_s_0+phase_corr),trimmed_width=im_diam)
    phi_ff = pup_s.copy()
    phi_ff = ff.retrieve_phase(im_S)
    phase_diversity = -gain*phi_ff.copy()
    print((phase_s_0+phase_corr)[pup_s==1].std())
    err.append((phase_s_0+phase_corr)[pup_s==1].std())
    phase_corr_old = phase_corr.copy()
plt.figure()
plt.plot(err,label='error')
plt.title('Phase error')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()

plt.matshow(phi_ff)
plt.matshow(phase_s_0+phase_corr)