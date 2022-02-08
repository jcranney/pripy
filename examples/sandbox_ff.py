import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from pripy import FastAndFurious
from tqdm import tqdm

# simulation parameters
pup_diam = 200
fourier_diam = 512
im_diam = 20

# build (circular) pupil
yy_s,xx_s = (np.mgrid[:pup_diam,:pup_diam]/(pup_diam-1)-0.5)
pup_s = 1.0*((yy_s**2+xx_s**2)**0.5 <= 0.5)

pad_by = ((fourier_diam-pup_diam)//2,(fourier_diam-pup_diam)//2)
pup_f = np.pad(pup_s,pad_by)

# offset to get the image centred on 2x2 pixels
phase_offset = (-(xx_s+yy_s)*(pup_diam-1)*2*np.pi/fourier_diam/2)

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

def trim_im(im,trimmed_width=im_diam):
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

# regularisation parameters
epsilon = np.pad(np.ones([im_diam,im_diam])*1e-5,(fourier_diam-im_diam)//2,mode='constant',constant_values=1e5)

# create Fast and Furious instance
ff = FastAndFurious(pup=pup_s,fft_width=pup_f.shape[0],
                    im_width=im_diam,offset=phase_offset,epsilon=epsilon)

# wavefront phase to be corrected (constant for this example):
phase_s_0      = 0.5*np.sin(xx_s*2*np.pi*3)+0.2*np.cos((xx_s+yy_s)*2*np.pi*2)

# correcting wavefront phase:
phase_corr     = phase_s_0 * 0.0
phase_corr_old = phase_s_0 * 0.0

# Fast and Furious integrator gain and leak factor
# n.b., the "diversity" introduced each iteration is a function of the gain
# and the leak factor. For a gain of 0.0 and a leak factor of 1.0, there would
# be no diversity, and therefore no resolution of the sign ambiguity of the even 
# wavefront components. 
gain = 0.01
leak = 0.99

# uncorrected wavefront phase:
plt.matshow(phase_s_0+phase_corr)

err = []
for i in tqdm(range(1000),leave=False):
    im_S = trim_im(phase_to_image(phase_s_0+phase_corr),trimmed_width=im_diam)
    phi_ff = ff.retrieve_phase(im_S)
    phase_diversity = -gain*phi_ff
    err.append((phase_s_0+phase_corr)[pup_s==1].std())
    phase_corr_old = phase_corr.copy()
    phase_corr *= leak
    phase_corr += phase_diversity
    ff.set_diversity_phase(phase_corr-phase_corr_old)

# error over time
plt.figure()
plt.plot(err,label='error')
plt.title('Phase error')
plt.xlabel('Iteration')
plt.ylabel('Error [rad RMS]')
plt.legend()

# residual wavefront phase:
plt.matshow(phase_s_0+phase_corr)