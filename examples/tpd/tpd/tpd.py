import torch as t
from torch import optim

from phase import image_from_phase, phase_from_zerns


default_opt = lambda x : opt.Adam(x, lr=0.1)

def tpd( alts, n_modes, imgs_target, alt_to_dir, *,
         psf_scaling = 1,optimizer_handle=default_opt,max_iter=5000):
    n_alt = alts.shape[0]
    x_alts = t.zeros((n_alt,n_modes),requires_grad=True,device=device)
    x_opt = t.zeros(x_alts.shape,requires_grad=True,device=device)
    optimizer = optimizer_handle(x_opt)
    sr  = 0
    exp_sr = 0.99
    loss_list = []
    sr_list   = []

    def cost(x,target):
        """evaluate cost function at guess 'x'

        """
        # compute images at each target
        dir_modes = t.cat([
          t.einsum("ijk,j->ik",alt_to_dir,x.flatten()),
          t.einsum("ijk,j->ik",alt_to_dir,(x+offset).flatten())
        ])
        imgs = image_from_phase(phase_from_zerns(dir_modes))/psf_scaling
        return t.cat([(imgs - target).flatten(),x.flatten()/10])


    for e in range(max_iter):
      loss = (t.square(cost(x_opt,imgs_target))).sum()
      loss_list.append(loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if(sr is not None):
        dir_modes = t.einsum("ijk,j->ik",alt_to_dir,(x_alts-x_opt).flatten())
        sr = t.mean((image_from_phase(phase_from_zerns(dir_modes))/psf_scaling).max(dim=1)[0].max(dim=1)[0]).detach().cpu().numpy()
        sr_list.append(sr)
        print("iteration {:5d} sr est: {:1.5f}".format(e,sr),end='\r')
        if(sr>exp_sr):
          print("\nTerminated with sr est > {:1.5f}".format(exp_sr))
          break
