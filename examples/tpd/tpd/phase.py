import numpy as np
import torch as t
import scipy.interpolate as interp
import aotools

as2rad = t.pi/180/3600

def phase_from_zerns(x: t.tensor, zerns : t.tensor):
    """get phase from zernike coefficients for a full pupil.

    Args:
        x : (np.ndarray): numpy array of coefficients shape of (max_zerns,)

        zerns : (np.ndarray) : array of zernike functions

    Returns:
        np.ndarray: phase (pup_diam,pup_diam)
    """
    if len(x.shape)==1:
        return t.einsum("ijk,i->jk",zerns,x)
    else:
        return t.einsum("ijk,...i->...jk",zerns,x)


def get_phase_to_zernike_projector(zerns : t.tensor, pup : t.tensor):
    """TODO
    """
    z = zerns[:,pup==1].T
    return t.linalg.solve(z.T @ z, z.T)

def create_phase_sample_coordinate(theta_x,theta_y,max_theta,alt,pup_diam_p,pup_diam_c,device='cpu'):
    yy,xx = np.mgrid[:pup_diam_p,:pup_diam_p]*1.0
    xx = t.tensor(xx.flatten(),device=device)
    yy = t.tensor(yy.flatten(),device=device)
    # scale down the sampling coordinates to correspond to the meta-pupil sub-region
    w = 1+2*t.abs(alt)*as2rad*max_theta/pup_diam_c
    xx /= w
    yy /= w
    # shift the coordinates according to altitude and field position
    sc = pup_diam_p*as2rad/pup_diam_c/w
    xx += (max_theta*t.abs(alt)+theta_x*alt)*sc
    yy += (max_theta*t.abs(alt)+theta_y*alt)*sc
    xx = xx.cpu().numpy()
    yy = yy.cpu().numpy()
    return xx,yy


def create_phase_interp_func(x,zerns,pup_diam_p):
    yy_og,xx_og = t.arange(pup_diam_p),t.arange(pup_diam_p)
    # build interpolator function
    # get phase of whole meta-pupil
    phi = phase_from_zerns(x,zerns).cpu().numpy()
    return interp.RegularGridInterpolator([xx_og,yy_og],phi,bounds_error=True)


def interpolate_phase_at_layer(interp_func,xx,yy,device='cpu'):
    pup_diam_p_x = interp_func.grid[0].max()+1
    pup_diam_p_y = interp_func.grid[1].max()+1
    # evaluate interpolator function at projected coordinates
    try:
        return t.tensor(interp_func(np.array([xx,yy]).T).reshape([pup_diam_p_x,pup_diam_p_y]),
                        dtype=t.float32,device=device)
    except ValueError:
        #print(xx.min(),xx.max(),yy.min(),yy.max())
        raise ValueError("Requested phase outside of defined metapupils")

def phase_at_layer(x: t.tensor, theta_x: float, theta_y: float, alt, zerns : t.tensor,
                   pup_diam_p, pup_diam_c, max_theta, *, device='cpu'):
    """compute the phase at a particular layer projected onto the pupil in a given direction

    Args:
        x (np.ndarray): numpy array of coefficients for the layer (max_zerns,)
        theta_x (float): x field position on-sky (arcseconds)
        theta_y (float): y field position on-sky (arcseconds)
        alt (float): layer altitude (metres)

    Returns:
        np.ndarray: phase (pup_diam,pup_diam)
    """
    # original coordinates (in pixels)
    interp_func = create_phase_interp_func(x,zerns,pup_diam_p)

    # define coordinates (in pixels) to sample that phase (projected from pupil)
    xx,yy = create_phase_sample_coordinate(theta_x,theta_y,max_theta,alt,pup_diam_p,pup_diam_c,
                                         device=device)

    ## evaluate interpolator function at projected coordinates
    return interpolate_phase_at_layer(interp_func,xx,yy)


def phase_in_direction(x: t.tensor, theta_x: float, theta_y: float, alts: t.tensor, zerns: t.tensor,
                       pup_diam_p, pup_diam_c, max_theta, pup):
    """calculate integral of phase from all layers in a particular direction

    Args:
        x (np.ndarray): coefficients for all layers (num_alts,max_zern)
        theta_x (float): x field position on-sky (arcseconds)
        theta_y (float): y field position on-sky (arcseconds)

    Returns:
        np.ndarray: phase (pup_diam,pup_diam)
    """
    device = pup.device
    phi = phase_at_layer(x[0],theta_x,theta_y,alts[0],zerns,pup_diam_p, pup_diam_c, max_theta,
                           device=device)
    for i in range(1,alts.size[0]):
        phi += phase_at_layer(x[i],theta_x,theta_y,alts[i],zerns,pup_diam_p, pup_diam_c, max_theta,
                           device=device)
    return phi*pup
    #return t.cat([phase_at_layer(xi,theta_x,theta_y,alt,zerns,pup_diam_p, pup_diam_c, max_theta,
    #                             device=device)[None,...]
    #              for xi,alt in zip(x,alts)]).sum(dim=0)*pup



def image_from_phase(phi: t.tensor, pup, *,wavelength: float = 0.55, window_size=32,
                     fft_width = 128):
    """compute images/PSFs given phases at pupil

    Args:
        phi (np.ndarray): phase in microns (n_phase,pup_diam_p,pup_diam_p)
        wavelength (float) or array-like: wavelength in microns

    Returns:
        np.ndarray: image (fft_width,fft_width)
    """
    only_one = False
    if len(phi.shape)==2:
        phi = phi[None,:,:]
        only_one = True
    im = t.abs(t.fft.fftshift(t.fft.fft2(
            pup[None,:,:]*t.exp(1j*2*t.pi*phi/wavelength),
            s=[fft_width,fft_width],norm="ortho",dim=[1,2]),dim=[1,2]
         ))**2
    im = im[:,im.shape[1]//2-window_size//2:im.shape[1]//2+window_size//2,
             im.shape[1]//2-window_size//2:im.shape[1]//2+window_size//2]
    if only_one == True:
        return im[0]
    else:
        return im