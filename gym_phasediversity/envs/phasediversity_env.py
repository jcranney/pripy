import gymnasium as gym
from gymnasium import error, spaces, utils, logger
from gymnasium.utils import seeding
from typing import Optional, Union

import numpy as np

class GaussDM:
    def __init__(self,pup_diam,nact_x,pitch,infl_coeff=0.3,valid_rad=None):
        actu_y,actu_x = np.mgrid[:nact_x,:nact_x]-nact_x/2+0.5
        actu_y,actu_x = [a.flatten() for a in [actu_y,actu_x]]
        yy,xx = np.mgrid[:pup_diam,:pup_diam]-pup_diam/2+0.5
        if valid_rad is not None:
            valid = (actu_x**2+actu_y**2)<(valid_rad**2)
            actu_y,actu_x = [a[valid] for a in [actu_y,actu_x]]
        actu_y,actu_x = [a*pitch for a in [actu_y,actu_x]]
        nactu = actu_x.shape[0]
        influs = np.zeros([nactu,pup_diam,pup_diam],dtype=np.float32)
        for influ,x,y in zip(influs,actu_x,actu_y):
            influ[:,:] = infl_coeff**(((x-xx)**2+(y-yy)**2)/pitch**2)
        self.influs = influs
        self.xx = actu_x
        self.yy = actu_y
    
    def dm_response(self,command):
        return np.einsum("ijk,i->jk",self.influs,command)


class PhaseDiversityEnv(gym.Env):
    """Environment for generic phase diversity.
    
    Observations:
        detector readout
    
    Actions:
        actuator voltages
    
    Rewards:
        reward = -rms(ideal_img-actual_img)*10
    
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    u_max = 100.0
    pup_width = 100
    nactu_x  = 19
    pitch = pup_width/(nactu_x-1)
    im_width = 40
    fft_width = 200 #384
    rebin_factor = 1
    sigma = 1.0
    buffer_length = 2
    
    def __init__(self):
        # initialise system
        self.nmeas  = self.im_width**2
        self.pup_s = self.make_pupil()
        self.dm = GaussDM(self.pup_width,self.nactu_x,pitch=self.pitch,valid_rad=(self.nactu_x-0.5)/2)
        self.nactu = self.dm.influs.shape[0]
        self.ideal = self.h_func(np.zeros(self.nactu,dtype=np.float32)).astype(np.float32)

        y_max = np.finfo(np.float32).max
        act_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(self.nactu,), dtype=np.float32)
        obs_space = spaces.Box(-y_max, y_max, shape=(self.im_width**2,), dtype=np.float32)
        self.action_space = act_space
        self.observation_space = spaces.flatten_space(spaces.Tuple(
            [act_space]*(self.buffer_length-1)+ \
            [obs_space]*(self.buffer_length)
        ))

        self.action_buffer = np.zeros([self.buffer_length-1,self.nactu],dtype=np.float32)
        self.state_buffer = np.zeros([self.buffer_length,self.im_width**2],dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.steps_beyond_done = None
        self._max_episode_steps = 20
    
    def get_phase(self,x):
        return self.dm.dm_response(x)

    def make_pupil(self):
        yy,xx = np.mgrid[:self.pup_width,:self.pup_width]-self.pup_width/2+0.5
        pup = (xx**2+yy**2)<((self.pup_width/2)**2)
        return pup
    
    def phase_to_image(self,phi):
        """Take wavefront phase in small pupil dimensions and return image
        
        Args:
            phi (ndarray): phase in small pupil dimensions

        Returns:
            ndarray: focal plane image
        """
        wf_s = self.pup_s * np.exp(1j*phi)
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
        return self.trim_im(self.phase_to_image(self.get_phase(x))).flatten()

    
    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        state_now = self.h_func(self._hidden_state+action).astype(np.float32)
        
        self.action_buffer = np.roll(self.action_buffer,[1],[0])
        self.action_buffer[0,:] = action

        self.state_buffer = np.roll(self.state_buffer,[1],[0])
        self.state_buffer[0,:] = state_now

        self.state = np.concatenate([self.state_buffer.flatten(),self.action_buffer.flatten()])

        done = not self.observation_space.contains(self.state)

        if not done:
            reward = state_now.max()**0.5
        elif self.steps_beyond_done is None:
            # just finished
            self.steps_beyond_done = 0
            reward = state_now.max()**0.5
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
        state_now = self.h_func(self._hidden_state).astype(np.float32)
        self.action_buffer *= 0.0
        self.state_buffer *= 0.0
        self.state_buffer[0,:] = state_now
        self.state = np.concatenate([self.state_buffer.flatten(),self.action_buffer.flatten()])
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
        background = pygame.surfarray.make_surface(np.tile(self.state[:self.im_width**2].reshape([self.im_width,self.im_width])[:,:,None],[1,1,3])**0.5*255)

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