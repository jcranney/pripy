#%%
import numpy as np
import torch as t
import matplotlib.pyplot as plt
plt.ion()
from  tpd.params import Params


target_n_alt    = 5
target_n_modes  = 100
target_alts     = t.linspace(-10000.0,10000,target_n_alt)
target_modes    = t.randn(*[target_n_alt,target_n_modes])*4e-3

# %% parameters
pup_diam_p  = 64  #!< pupil diameter in pixel
pup_diam_m  = 8   #!< pupil diameter in meter
n_zernike   = 100 #!< number of modes to retrieve
theta       = 30  #!< camera angle in arcsec
alts        = t.linspace(-10000.0,10000,5) #!< conjugation altitudes
window_size = 64
fft_width   = 128

defocus     = [0,0.1]

# %% sources
n_src_x     = 8

max_theta   = theta/2*2**0.5  # including the rotation
pos_x,pos_y = t.meshgrid(t.arange(n_src_x),t.arange(n_src_x),indexing="ij")
pos_x = (pos_x-(n_src_x-1)/2)/(n_src_x-1)*theta
pos_y = (pos_y-(n_src_x-1)/2)/(n_src_x-1)*theta
pos_x = pos_x.flatten()
pos_y = pos_y.flatten()

# %%
n = target_modes.size()[1]
target = Params(pup_diam_p, pup_diam_m, max_theta, n, target_alts, pos_x, pos_y)
offset     = target.get_offsets(defocus,symmetric=True)
target_img = target.get_images(target_modes,window_size,offset = offset)

#%%
fig,ax = plt.subplots(n_src_x*offset.size()[0],n_src_x,figsize=[8,16])
for i,im in enumerate(target_img):
    ax.flatten()[i].imshow(im.cpu())
    ax.flatten()[i].set_xticks([])
    ax.flatten()[i].set_yticks([])
plt.tight_layout()

# %%
tpd = Params(pup_diam_p, pup_diam_m, max_theta, n, alts, pos_x, pos_y)

# %%
max_iter=100
print("using CPU")
tpd.set_device('cpu')
x_cpu,data_cpu = tpd.find_x(target_img, offset = offset,max_iter=max_iter)
sr_cpu = t.mean(
            tpd.diff_modes_images(
                x_cpu-target_modes,target_img,offset=offset,
            ).max(dim=1)[0].max(dim=1)[0]).detach().cpu().numpy()
print("SR:",sr_cpu)

# %%
tpd.set_device('cuda:7')
print("using GPU")
x_gpu,data_gpu = tpd.find_x(target_img, offset = offset,max_iter=max_iter)
sr_gpu = t.mean(
            tpd.diff_modes_images(
                x_gpu-target_modes.to(tpd.device),target_img.to(tpd.device),offset=offset.to(tpd.device),
            ).max(dim=1)[0].max(dim=1)[0]).detach().cpu().numpy()
print("SR:",sr_gpu)

#%%
plt.plot(target_modes.flatten(),color='black',label="modes target")
plt.plot(x_cpu.flatten(),label="modes CPU")
plt.plot(x_gpu.flatten().cpu(),label="modes GPU")
# %%
plt.figure()
plt.plot((target_modes-x_cpu).flatten(),label="CPU")
plt.plot((target_modes-x_gpu.cpu()).flatten(),label="GPU")
print(t.abs(x_cpu-x_gpu.cpu()).max())
#%%
plt.plot(data_cpu[1],label="loss CPU")
plt.plot(data_gpu[1],label='loss GPU')
# %%
