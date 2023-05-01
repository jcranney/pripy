import torch as t
from torch import optim
from tqdm import tqdm
import aotools
from . import phase
import sys

class Params:
    def __init__(self, pup_diam_p, pup_diam_m, max_theta, n_modes, alts, pos_x, pos_y,device='cpu'):
        self._device    = device
        self.pup_diam_p = pup_diam_p
        self.pup_diam_c = pup_diam_m
        self.max_theta  = max_theta
        # array of zernike functions
        self.zerns = t.tensor(aotools.zernikeArray(n_modes,pup_diam_p),dtype=t.float32,device=device)
        self.pup   = t.tensor(aotools.phaseFromZernikes([1],pup_diam_p),dtype=t.float32,device=device)

        z = self.zerns[:,self.pup==1].T
        self.phase_to_zernike = t.linalg.solve(z.T @ z, z.T)

        self.projector_alt_to_dir = None
        self.set(alts = alts, n_modes=n_modes, pos_x=pos_x, pos_y=pos_y)

        self._dir_modes = None

        self.psf_scaling = phase.image_from_phase(self.pup*0.0,self.zerns).max()

    def set_device(self,device):
        self.zerns = self.zerns.to(device=device)
        self.pup = self.pup.to(device=device)
        self.projector_alt_to_dir = self.projector_alt_to_dir.to(device=device)
        self._device = device

    @property
    def device(self):
        return self._device

    @property
    def n_alts(self):
        return self.alts.size()[0]

    @property
    def n_modes(self):
        return self._n_modes

    def phase_at_layer(self,x, theta_x: float, theta_y: float, *, layer_id: int = None):
        if layer_id is not None:
            phi = phase.phase_at_layer(x, theta_x, theta_y,self.alts[layer_id], self.zerns,
                                        self.pup_diam_p, self.pup_diam_c, self.max_theta,
                                        device=self.device)
        else:
            phi = t.zeros(self.n_alts,self.pup_diam_p,self.pup_diam_p,device=self.device)
            for i in range(self.n_alts):
                phi[i] = self.phase_at_layer(x[i],theta_x, theta_y, layer_id=i)
        return phi

    def phase_in_direction(self,x: t.tensor, theta_x: float, theta_y):
        #return t.cat(self.phase_at_layer(x,theta_x, theta_y)).sum(dim=0)*self.pup
        return self.phase_at_layer(x,theta_x, theta_y).sum(dim=0)*self.pup


    def _create_projector_alt_to_dir(self):
        self.projector_alt_to_dir = t.zeros(self.pos_x.size()[0],self.n_alts*self.n_modes,self.n_modes,
                                            device=self.device)
        x_tmp = t.zeros([self.n_modes])
        for i in tqdm(range(self.n_modes)):
            x_tmp[i] = 1.0
            interp_func = phase.create_phase_interp_func(x_tmp.to(self.device),self.zerns,
                                                         self.pup_diam_p)
            x_tmp[i] = 0.0
            for ell in range(self.n_alts):
                for k, (px,py) in enumerate(list(zip(self.pos_x, self.pos_y))):
                    xx,yy = phase.create_phase_sample_coordinate(px,py,self.max_theta,
                                self.alts[ell],self.pup_diam_p,self.pup_diam_c,device=self.device)
                    phi = phase.interpolate_phase_at_layer(interp_func,xx,yy)[self.pup==1]
                    self.projector_alt_to_dir[k,ell*self.n_modes+i] = \
                        self.phase_to_zernike @ phi.to(self.device)


    def set(self, *, alts=None, n_modes=None, pos_x=None,pos_y=None):
        if pos_x is not None or pos_y is not None:
            if pos_x.shape != pos_y.shape:
                raise RuntimeError("pos_x and pos_y must have the same shape")
            self.pos_x = pos_x.to(self.device)
            self.pos_y = pos_y.to(self.device)
        if alts is not None:
            if not isinstance(alts,t.Tensor):
                alts = t.tensor(alts)
            self.alts = alts.to(self.device)
        if n_modes is not None:
            self._n_modes = n_modes
        self._create_projector_alt_to_dir()

    def get_offsets(self, defocus, *, symmetric=False, device=None):
        if device is None:
            device = self.device
        if symmetric :
            defocus += [-d for d in defocus]
        tmp = []
        for x in defocus:
            if x not in tmp:
                tmp.append(x)
        defocus = tmp

        n_defoc = len(defocus)
        offset = t.zeros(n_defoc,self.n_alts,self.n_modes, device = device)
        for i in range(len(defocus)):
            offset[i, self.alts==0, 3] = defocus[i]
        return offset


    def get_images(self, x, window_size, *, offset=None, wavelength: float=0.55,
                   fft_width = 128):
        if offset is None:
            offset = t.zeros(1,*x.size(),device=self.device)
        if offset.ndim ==2 :
            offset = offset[None,...].to(self.device)
        imgs = phase.image_from_phase(
                    phase.phase_from_zerns(
                        t.einsum("ijk,lj->lik",self.projector_alt_to_dir,
                                               (offset + x).flatten(1,2)
                                ).flatten(0,1),
                        self.zerns
                    ),self.pup,
                    window_size=window_size, wavelength=wavelength,fft_width=fft_width)
        return imgs/ self.psf_scaling


    def diff_modes_images(self, x, images, *, offset=None, wavelength: float=0.55,
                   fft_width = 128):
        window_size = images.size()[1]
        return self.get_images(x, window_size, offset=offset, wavelength=wavelength,
                               fft_width=fft_width) - images

    def cost(self,x, images, window_size,offset, wavelength, fft_width):
        diff = self.diff_modes_images(x, images, offset=offset, wavelength=wavelength,
                                      fft_width=fft_width)
        return t.cat([diff.flatten(), x.flatten()/10])

    def find_x(self, images, *, x=None, offset=None, wavelength: float=0.55,
               fft_width = 128, real_x=None,max_iter=5000):

        images = images.to(self.device)
        window_size = images.size()[1]
        if x is None:
            x = t.zeros(self.n_alts,self.n_modes,requires_grad=True,device=self.device)
        else:
            x = x.to(self.device)

        if offset is None:
            offset = [0]
        if isinstance(offset,list):
            offset = self.get_offsets(offset)
        else:
          offset = offset.to(self.device)

        if real_x is not None:
            real_x = real_x.to(self.device)

        optimizer = optim.Adam([x], lr=1e-3)
        sr  =0
        exp_sr = 0.99
        loss_list = []
        sr_list   = []
        for e in tqdm(range(max_iter)):
            loss = (t.square(
                      self.cost(x, images,window_size, offset,wavelength,fft_width)
                            )).sum()
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if real_x is not None:
              imgs = self.get_images(real_x-x, window_size, wavelength=wavelength,
                                     fft_width=fft_width)
              sr = t.mean(imgs.max(dim=1)[0].max(dim=1)[0]).detach().cpu().numpy()
              #tqdm.write("iteration {:5d} sr est: {:1.5f}\r".format(e,sr),file=sys.stdout)
            if(sr>exp_sr):
              print("\nTerminated with sr est > {:1.5f}".format(exp_sr))
              break
        return x.detach(),(optimizer,loss_list,sr_list)