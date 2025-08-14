try:
    import cupy as cp  # type: ignore
except ImportError:
    Warning("No cupy available, using basic numpy")
    import numpy as cp

import numpy as np

import scipy.optimize as opt
from collections.abc import Callable
from segment_phasing_fp_env.psf import PSF


class FastAndFurious:
    """Fast and Furious is a class for computing the phase of a wavefront
    from an image using sequential phase diversity. It is based on the
    algorithm described in Korkiakoski et al, Appl. Opt. 53, 4565-4579 (2014).

    Parameters
    ----------
    pup_small : ndarray
        The pupil function.
    im_width : int
        The width of the image.
    fft_width : int
        The width of the FFT.
    epsilon : float/ndarray (fft_width x fft_width)
        The regularisation parameter used in the focal plane of the FF
        algorithm. Can be a scalar or an array in the fft domain. E.g., this
        parameter should be set to a large number if the corresponding pixel
        is noisy/unused. See example for details.
    image_center : str
        The centering mode for the image. Can be either "image_center"
        (default) or "fft_center", depending on whether the image is centered
        on one or two pixels

    Attributes
    ----------
    pup_width : int
        The width of the pupil plane arrays.
    im_width : int
        The width of the focal plane arrays.
    fft_width : int
        The width of the FFT support.

    Methods
    -------
    compute_phase(p_i)
        Compute the phase of the wavefront from an image.
    set_diversity_phase(dphi)
        Set the diversity phase.

    ### References
    # Bos et al., On-sky verification of Fast and Furious focal-plane wavefront
    #  sensing: Moving forward toward controlling the island effect at
    # Subaru/SCExAO https://doi.org/10.1051/0004-6361/202037910
    # (we will use the equation numbers from the paper)

    # Original paper is harder to follow and also has an error in one of the
    # equations: Keller et al, Extremely fast focal-plane wavefront sensing for
    # extreme adaptive optics, https://arxiv.org/pdf/1207.3273.pdf
    """

    def __init__(
        self,
        pup: np.ndarray,
        im_width: int,
        fft_width: int,
        epsilon: float = 1e-5,
        image_center: str = "image_center",
    ):
        """Initialise an instance of FastAndFurious."""
        pup_width = pup.shape[0]  # width of pupil
        self.im_width = im_width  # width of image
        self.pup_width = pup_width  # width of the pupil
        self.fft_width = fft_width  # width of fft

        if image_center == "image_center":
            print(
                "Assuming that the image is centered on 2x2 pixels at the"
                " center of the array"
            )
            self.image_center = "image_center"
            # centering the image on 2x2 pixels:
            yy_s, xx_s = (
                np.mgrid[:pup_width, :pup_width] / (pup_width - 1) - 0.5
            )
            offset = (
                -(xx_s + yy_s) * (pup_width - 1) * 2 * np.pi / fft_width / 2
            )
        else:
            print(
                "Assuming that the image is centered on 1 pixel at [N/2,N/2]"
            )
            self.image_center = "fft_center"
            offset = cp.zeros((pup_width, pup_width))

        self._offset = cp.exp(
            1j
            * cp.pad(cp.array(offset), (self.fft_width - self.pup_width) // 2)
        )

        # Set up the (hidden) parameters. These are all either scalars or cupy
        # arrays. Pupil in FFT dimensions:
        self._pup = cp.pad(cp.array(pup), (self.fft_width - pup.shape[0]) // 2)

        # Focal plane complex amplitude (real valued -- assuming symmetry in
        # pupil):
        self._a = (
            cp.fft.fftshift(
                cp.fft.fft2(cp.fft.fftshift(self._pup * self._offset))
            )
        ).real / self._pup.sum()  # focal plane amplitude
        self._a2 = self._a**2  # focal plane intensity

        # Constants:
        self._epsilon = cp.array(epsilon)  # regularization parameter
        self._ydenom = (
            2 * self._a2 + self._epsilon
        )  # Eq. (5) of Bos (Eq (25) of Keller is a mistake)
        self._S = 1.0
        self.reset()

    def reset(self):
        """Reset the FF algorithm/recursive parameters"""
        self._v_d = None
        self._y_d = None
        self._p_de = None
        self._reset = True

    def compute_phase(self, p_i):
        """Compute the phase of the wavefront from an image.

        Note that the phase in all variables within FF is expected to be in
        radians, so typically one will need to scale returned phase by
        lambda/(2*pi) after calling this function.

        Parameters
        ----------
        p_i : ndarray (im_width x im_width)
            The image used for wavefront retrieval.

        Returns
        -------
        ndarray (pup_width x pup_width)
            The estimated phase of the wavefront.
        """

        # pad to the fft size and rescale:
        p_i = cp.pad(cp.array(p_i), (self.fft_width - self.im_width) // 2)
        p_i *= self._a2.sum() / p_i.sum()

        # odd/even decomposition:
        if self.image_center == "image_center":
            p_io = 0.5 * (p_i - p_i[::-1, ::-1])  # odd component of image
            p_ie = 0.5 * (p_i + p_i[::-1, ::-1])  # even component of image
        else:
            p_io = 0.5 * (
                p_i - np.roll(p_i[::-1, ::-1], (1, 1), axis=(0, 1))
            )  # odd component of image
            p_ie = 0.5 * (
                p_i + np.roll(p_i[::-1, ::-1], (1, 1), axis=(0, 1))
            )  # even component of image

        # compute imaginary part of FFT(phase):
        y_i = self._a * p_io / self._ydenom  # Eq (5) of Bos

        # compute sign of real part of FFT(phase):
        if self._reset:
            self._v_sign = cp.sign(self._a.copy())
            self._reset = False
        elif self._v_d is not None and self._y_d is not None:
            self._v_sign = cp.sign(
                self._p_de
                - p_ie
                - (self._v_d**2 + self._y_d**2 + 2 * y_i * self._y_d)
            ) * cp.sign(
                self._v_d
            )  # Eq (9) of Bos, modified to avoid div by zero.

        # compute abs(real part of FFT(phase)):
        v_i_abs = cp.sqrt(
            cp.abs(p_ie - (self._S * self._a2 + y_i**2))
        )  # Eq. (6) of Bos

        # combine to get real part of FFT(phase):
        v_i = v_i_abs * self._v_sign

        # inverse transform to get phase + negative v_i (not sure why!)
        scaling_constant = self._pup.sum() / (
            self._pup.shape[0] * self._pup.shape[1]
        )
        # this value "works" for a range of pup_width and fft_width but it
        # is a guess
        phi = (
            scaling_constant
            * cp.fft.fftshift(
                cp.fft.ifft2(cp.fft.fftshift(-v_i - 1j * y_i), norm="forward")
            ).real
        )  # Eq. (10) of Bos

        # save even part of image for next iteration:
        self._p_de = p_ie.copy()

        # return phase estimate on pupil plane support:
        phi = phi[
            self.fft_width // 2 - self.pup_width // 2: self.fft_width // 2
            + self.pup_width // 2,
            self.fft_width // 2 - self.pup_width // 2: self.fft_width // 2
            + self.pup_width // 2,
        ]

        # return numpy array regardless of if using cupy or not:
        if hasattr(phi, "get"):
            return phi.get()
        else:
            return phi

    def set_diversity_phase(self, dphi):
        """Set the diversity phase. This should be the difference between the
        most recent phase and the phase of the wavefront at the previous image.
        I.e., the difference between the two DM shapes, assuming the turbulence
        change is negligble.

        Note that the phase in all variables within FF is expected to be in
        radians, so typically one will need to scale the phase by 2*pi/lambda
        before passing to this function.

        Parameters
        ----------
        dphi : ndarray (pup_width x pup_width)
            The diversity phase.
        """
        dphi_big = cp.pad(
            cp.array(dphi), (self.fft_width - self.pup_width) // 2
        )
        dphi_big_o = (dphi_big - dphi_big[::-1, ::-1]) * 0.5
        dphi_big_e = (dphi_big + dphi_big[::-1, ::-1]) * 0.5
        self._y_d = cp.fft.fftshift(
            cp.fft.ifft2(cp.fft.fftshift(dphi_big_o))
        ).imag
        self._v_d = cp.fft.fftshift(
            cp.fft.ifft2(cp.fft.fftshift(dphi_big_e))
        ).real


class MHE:
    """Moving Horizon Estimator"""

    def __init__(
        self,
        nstate: int,
        nmeas: int,
        nbuffer: int,
        noise_cov: np.ndarray,
        state_cov: np.ndarray,
        state_matrix: np.ndarray,
        h_eval: Callable,
    ):
        """Initialise the estimator"""
        self._nstate = nstate  # len(x)
        self._nmeas = nmeas  # len(y)
        self._nbuffer = nbuffer  # mhe horizon length
        self._h_eval = h_eval  # h(x)
        process_cov = state_cov - state_matrix @ state_cov @ state_matrix.T

        state_cov_inv = np.linalg.inv(state_cov)
        process_cov_inv = np.linalg.inv(process_cov)

        gamma = np.zeros([nstate * nbuffer, nstate * nbuffer])
        gamma[:nstate, :nstate] = state_cov_inv - process_cov_inv
        gamma[-nstate:, -nstate:] = (
            -state_matrix.T @ process_cov_inv @ state_matrix
        )
        for mi in range(nbuffer):
            for ni in range(nbuffer):
                if mi == ni:
                    tmp = (
                        state_matrix.T @ process_cov_inv @ state_matrix
                        + process_cov_inv
                    )
                    gamma[
                        mi * nstate: (mi + 1) * nstate,
                        ni * nstate: (ni + 1) * nstate,
                    ] += tmp
                elif mi == (ni + 1):
                    tmp = -process_cov_inv @ state_matrix
                    gamma[
                        mi * nstate: (mi + 1) * nstate,
                        ni * nstate: (ni + 1) * nstate,
                    ] = tmp
                elif mi == (ni - 1):
                    tmp = -state_matrix.T @ process_cov_inv
                    gamma[
                        mi * nstate: (mi + 1) * nstate,
                        ni * nstate: (ni + 1) * nstate,
                    ] = tmp
        self._gamma_factor = np.linalg.cholesky(gamma).T  # maybe no transpose?
        noise_cov_inv = np.linalg.inv(noise_cov)
        self._noise_cov_inv_factor = np.linalg.cholesky(
            noise_cov_inv
        ).T  # maybe no transpose?

    def cost_vector(self, x, x_dm, yd):
        """Evaluate MHE cost vector (vector of residuals to be square-summed
        for actual cost function, or fed into a "least-squares" optimiser for
        state, x, measurements, yd, and dm sequence, x_dm.
        """
        h = np.r_[
            [
                self._h_eval(
                    (x + x_dm)[self._nstate * i: self._nstate * (i + 1)]
                )
                for i in range(self._nbuffer)
            ]
        ]
        cost = np.r_[
            [self._noise_cov_inv_factor @ (hi - ydi) for hi, ydi in zip(h, yd)]
        ].flatten()
        cost = np.r_[cost, self._gamma_factor @ x]
        return cost

    def get_estimate(self, x0, x_dm, yd):
        """Get the current estimate of the state based on recent priors.

        Args:
            x0 (np.ndarray): Initial state estimate (best guess)
            x_dm (np.ndarray): dm sequence (N,NSTATE)
            yd (np.ndarray): Measurement sequence (N,NMEAS)

        Returns:
            np.ndarray: Current state estimate
        """
        xopt = opt.least_squares(
            lambda x: self.cost_vector(x, x_dm, yd),
            x0,
        )
        return xopt["x"][-self._nstate:]

    @staticmethod
    def from_model(model: PSF, *, nbuffer: int = 10):
        nmeas = model.image.flatten().shape[0]
        nstate = model.nmodes
        noise_cov = np.eye(nmeas) * model.noise**2
        state_cov = np.eye(nstate) * model.sigma**2
        state_matrix = np.eye(nstate) * model.corr
        return MHE(
            nstate=nstate,
            nmeas=nmeas,
            nbuffer=nbuffer,
            noise_cov=noise_cov,
            state_cov=state_cov,
            state_matrix=state_matrix,
            h_eval=lambda x: model.poke(x).flatten(),
        )


class GerchbergSaxton:
    """Gerchberg-Saxton algorithm for estimating the phase of a wavefront given
    the pupil mask and focal-plane image.

    Args:
        pup : (np.ndarray)
            Pupil mask, will be padded to be fft_width x fft_width.
        wavelength : (float)
            Wavelength of light, in whatever units you like. Retrieved phase
            will be in the same units.
        fft_width : (int)
            Width of the FFT support in pixels.
        im_width : (int)
            Width of the image in pixels.
        offset : (np.ndarray)
            implicit constant phase offset (in radians), e.g. to centre image
            on a quadcell (2x2 pixels), rather than a single pixel (default).

    Methods:
        compute_phase : Compute the phase of the wavefront.
        reset : Reset the init_phase for the GS iterations.


    WARNING: Incomplete implementation.
    Requires:
     - binning considerations.
    """

    def __init__(self, pup, wavelength, fft_width, im_width, offset=None):
        # the image, pupil, and fft support need not be the same size, though
        # im_width and pup_width must be no larger than the fft_width.
        self._im_width = im_width
        self._fft_width = fft_width
        self._pup_width = pup.shape[0]

        # pad the pupil mask to be in the fft support
        pup_big = cp.pad(cp.array(pup), (self._fft_width - pup.shape[0]) // 2)
        # fft shift pupil so we can do less fft-shifts each iteration
        self._pup_shft = cp.fft.fftshift(pup_big)

        # user-exposed wavefront phase will always be in the units inferred by
        # wavelength (except for the offset phase, which is in radians). E.g.,
        # gs.compute_phase(im) returns wavefront phase in wavelength units.
        self._wavelength = wavelength

        # scaling factor for image normalisation
        self._scf = (cp.abs(cp.fft.fft2(self._pup_shft)) ** 2).sum()

        # user-supplied offset is padded and fft-shifted, to be applied before
        # and removed after gs algorithm (such that it is invisible to the
        # user)
        if offset is not None:
            self._offset_shft = cp.fft.fftshift(
                cp.pad(
                    cp.array(offset), (self._fft_width - offset.shape[0]) // 2
                )
            )
        else:
            self._offset_shft = None
        # assume that the pixels outside the imager are invalid. This attribute
        # can be set by the user if a different "invalid_pixel" range is
        # desired.
        self.invalid_pixels = cp.fft.fftshift(
            cp.pad(
                cp.ones([self._im_width, self._im_width]),
                (self._fft_width - self._im_width) // 2,
            )
            == 0
        )

        # init phase to be used if not provided by user in compute_phase() func
        self.reset()

    def reset(self, init_phase=None):
        """
        Reset the init_phase for the GS iterations.

        Args:
            init_phase : np.ndarray
                array to be used as the initialisation phase the next time
                compute_phase() is called. If None, will be randomised.
        """
        if init_phase is None:
            # by default, use a random initialisation
            self._phasepup_shft = cp.random.randn(*self._pup_shft.shape)
        else:
            # init is provided, so convert to cp.ndarray
            # (recalling that cupy === numpy if cupy isn't available)
            if type(init_phase) is not cp.ndarray:
                init_phase = cp.array(init_phase)
            # assume init was provided in wavelength units, so convert to
            # radians
            init_phase = init_phase * (2 * cp.pi / self._wavelength)

            # if init is not big enough, pad it with the same convention as
            # the pupil padding
            if not cp.all(cp.r_[init_phase.shape] == self._fft_width):
                init_phase = cp.pad(
                    init_phase, (self._fft_width - init_phase.shape[0]) // 2
                )
            # fftshift to be self consistent.
            init_phase = cp.fft.fftshift(init_phase)

            # if an offset was provided in object creation, apply it to the
            # init phase
            if self._offset_shft is not None:
                init_phase += self._offset_shft

            # copy the new init to the init_phase attribute to be used in
            # _gs_run
            self._phasepup_shft = init_phase.copy()

    def _rebin(self, a, newshape):
        """
        Rebin an array to a new shape.

        Args:
            a (np.ndarray): Array to be rebinned
            newshape (tuple): New shape of the array

        Returns:
            np.ndarray: Rebinned array
        """
        slices = [
            slice(None, None, old / new) for old, new in zip(a.shape, newshape)
        ]
        return a[slices]

    def _gs_run(
        self, amp_shft, iterations=20, hio_param=0.3, invalid_pixels=None
    ):
        """Run the GS algorithm on the input values.

        Args:

            amp_shft : (cp.ndarray)
                Focal-plane amplitude after fft-shifting (fft_width,fft_width).
            iterations : (int)
                Number of iterations to run.
            hio_param : (float)
                Parameter for HIO step.
            invalid_pixels : (cp.ndarray)
                Invalid pixels (fft_width,fft_width). If not None, will be used
                to mask out invalid pixels in focal-plane.

        Returns:
            phasepup_shft : (cp.ndarray)
                Phase estimate from GS iterations (fft_width,fft_width)
        """

        amppup_shft = self._pup_shft.copy()
        for _ in range(iterations):
            cplxpup_shft = (
                (1.0 + hio_param) * self._pup_shft - hio_param * amppup_shft
            ) * cp.exp(1j * self._phasepup_shft)
            cplxim_shft = cp.fft.fft2(cplxpup_shft)

            phaseim_shft = cp.angle(cplxim_shft)
            ampim_shft = cp.abs(cplxim_shft)

            cplxim_shft2 = (
                (1.0 + hio_param) * amp_shft - hio_param * ampim_shft
            ) * cp.exp(1j * phaseim_shft)
            if invalid_pixels is not None:
                cplxim_shft2[invalid_pixels] = cplxim_shft[
                    invalid_pixels
                ].copy()

            cplxpup_shft2 = cp.fft.ifft2(cplxim_shft2)
            self._phasepup_shft = cp.angle(cplxpup_shft2)
            amppup_shft = cp.abs(cplxpup_shft2)

        # save the image computed from the GS estimated phase (for debugging):
        self._gs_im = (cp.fft.fftshift(ampim_shft) ** 2) * (  # type: ignore
            self._imsum / self._scf
        )
        return self._phasepup_shft.copy()

    def compute_phase(
        self, im, iters=100, hio_param=0.3, discard_invalid=True
    ):
        """Compute and return the phase estimated by the Gerchberg-Saxton
        algorithm, given an image.

        Args:
            im (np.ndarray): Image to perform GS on.
            iters (int): Number of iterations of GS to perform.
            hio_param (float): Parameter for the Hybrid Input-Output (HIO)
                step.
            discard_invalid (bool): If True, discard invalid pixels in WFS
                image.

        Returns:
            np.ndarray: Phase estimate (in wavelength units).
        """
        self._imsum = im.sum()
        im = cp.array(im) * (self._scf / self._imsum)
        im = cp.pad(im, (self._fft_width - self._im_width) // 2)
        im = im**0.5
        im_shft = cp.fft.fftshift(im)

        # check if user wants to discard invalid pixels this time
        if discard_invalid:
            invalid_pixels = self.invalid_pixels
        else:
            invalid_pixels = None

        # perform gs algo
        out_phi_shft = self._gs_run(
            im_shft,
            iterations=iters,
            hio_param=hio_param,
            invalid_pixels=invalid_pixels,
        )

        # if an offset was provided in object creation, remove it from the GS
        # estimated phase.
        if self._offset_shft is not None:
            out_phi_shft -= self._offset_shft

        # inverse fftshift and trim to pupil size
        out_phi = cp.fft.ifftshift(out_phi_shft)[
            (self._fft_width - self._pup_width) // 2: (
                self._fft_width + self._pup_width
            )
            // 2,
            (self._fft_width - self._pup_width) // 2: (
                self._fft_width + self._pup_width
            )
            // 2,
        ]

        # scale output phase to be in wavelength units
        out_phi *= self._wavelength / (2 * cp.pi)

        # return numpy array. Recall that cupy might just be an alias for numpy
        # if cupy is not available, so have to check if cp.array has a "get"
        # method.
        if hasattr(out_phi, "get"):
            return out_phi.get()
        else:
            return out_phi
