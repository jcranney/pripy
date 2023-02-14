import gym
from gym import error, spaces, utils
from gym.utils import seeding
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.utils import seeding

class GMTPhasingEnv(gym.Env):
    """Environment for GMT segment phasing.
    
    Observations:
        detector readout
    
    Actions:
        actuator voltages
    
    Rewards:
        reward = -rms(ideal_img-actual_img)*10
    
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    u_max = 10.0
    pup_width = 100
    im_width = 28
    fft_width = 384
    rebin_factor = 2
    nactu  = 9
    gap_width_m = 0.359
    seg_width_m = 8.4
    sigma = 1.0
    alpha = 0.999
    beta = sigma*(1-alpha**2)**0.5

    def __init__(self):
        # initialise system
        self.x0 = self.pup_width/2
        self.y0 = self.pup_width/2
        self.nmeas  = self.im_width**2
        self.pup_s = self.make_pupil(self.x0,self.y0)
        self.phimat = np.concatenate([
            self.make_tt_phimat(self.x0,self.y0),
            self.make_phimat(self.x0,self.y0)
        ])
        # offset to get the image centred on 2x2 pixels
        yy_s,xx_s = np.meshgrid(np.arange(self.pup_width)/(self.pup_width-1)-0.5,
                               np.arange(self.pup_width)/(self.pup_width-1)-0.5)
        self.phase_offset = -(xx_s+yy_s)*(self.pup_width-1)*2*np.pi/self.fft_width/2

        self.ideal = self.h_func(np.zeros(self.nactu,dtype=np.float32)).astype(np.float32)

        y_max = np.finfo(np.float32).max
        self.action_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(self.nactu,), dtype=np.float32)
        self.observation_space = spaces.Box(-y_max, y_max, shape=(self.im_width, self.im_width), dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.steps_beyond_done = None
    
    def get_phase(self,x):
        return np.einsum("ijk,i->jk",self.phimat,x)

    def make_pupil(self,x0,y0):
        return self.make_phimat(x0,y0).sum(axis=0)

    def make_phimat(self,x0,y0):
        seg_diam = self.pup_width/(3+self.gap_width_m/self.seg_width_m*2)
        yy,xx = np.mgrid[:self.pup_width,:self.pup_width]*1.0
        zz = np.tile(xx[None,:,:]*0,[7,1,1])
        zz[0,:,:] = (((xx-x0)**2+(yy-y0)**2)**0.5 < seg_diam/2)*1.0
        for i,theta in enumerate(np.linspace(0,2*np.pi,7)[:-1]):
            radial_offset = seg_diam*(1+self.gap_width_m/self.seg_width_m)
            zz[1+i,:,:] = (((xx-x0-radial_offset*np.cos(theta))**2+(yy-y0-radial_offset*np.sin(theta))**2)**0.5 < seg_diam/2)*1.0
        return zz

    def make_tt_phimat(self,x0,y0):
        yy,xx = np.mgrid[:self.pup_width,:self.pup_width]*1.0
        xx -= x0
        yy -= y0
        zz = np.tile(xx[None,:,:]*0,[2,1,1])
        zz[0,:,:] = xx/self.pup_width
        zz[1,:,:] = yy/self.pup_width
        return zz

    def phase_to_image(self,phi):
        """Take wavefront phase in small pupil dimensions and return image
        
        Args:
            phi (ndarray): phase in small pupil dimensions

        Returns:
            ndarray: focal plane image
        """
        wf_s = self.pup_s * np.exp(1j*(phi+self.phase_offset))
        im_F = np.abs(np.fft.fftshift(np.fft.fft2(wf_s,s=[self.fft_width,self.fft_width]))/self.pup_s.sum())**2
        return im_F

    @staticmethod
    def rebin(a, factor):
        sh = a.shape[0]//factor,factor,a.shape[1]//factor,factor
        return a.reshape(sh).mean(-1).mean(1)

    def trim_im(self,im):
        """Trim image to a square of size trimmed_width x trimmed_width

        Args:
            im (ndarray): image to trim
            trimmed_width (int): size of the trimmed image

        Returns:
            ndarray: trimmed image
        """
        im = self.rebin(im,self.rebin_factor)
        return im[
            im.shape[0]//2-self.im_width//2:im.shape[0]//2+self.im_width//2,
            im.shape[1]//2-self.im_width//2:im.shape[1]//2+self.im_width//2
            ]

    def h_func(self,x):
        "takes state, returns image"
        return self.trim_im(self.phase_to_image(self.get_phase(x)))

    
    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        self._hidden_state *= self.alpha
        self._hidden_state += action
        self._hidden_state += self.beta*np.random.randn(self.nactu)
        self.state = self.h_func(self._hidden_state).astype(np.float32)

        done = not self.observation_space.contains(self.state)

        if not done:
            reward = -((self.state-self.ideal)**2).sum()**0.5*10
        elif self.steps_beyond_done is None:
            # just finished
            self.steps_beyond_done = 0
            reward = -((self.state-self.ideal)**2).sum()**0.5*10
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
        options: Optional[dict] = None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self._hidden_state = np.random.randn(self.nactu)*self.sigma
        self.state = self.h_func(self._hidden_state).astype(np.float32)
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
    
    def render(self, mode='human'):
        import pygame
        # create a surface on screen
        win_width = 400
        win_height = 400
        background = pygame.surfarray.make_surface(np.tile(self.state[:,:,None],[1,1,3])**0.5*255)

        if self.state is None:
            return None

        if self.screen is None:
            # initialize the pygame module
            pygame.init()
            pygame.display.set_caption(f"WFS Image")
            self.screen = pygame.display.set_mode((win_width,win_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        background = pygame.transform.scale(background, self.screen.get_size())
        
        self.screen.blit(background,(0,0))
        
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    
        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False