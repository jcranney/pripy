import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from matplotlib.animation import FuncAnimation
import seaborn
seaborn.set_theme()


frames = np.load("frames_gym.npy")[:]
frames[frames<0] = 0
frames = frames**0.5
clim = [frames.min(),frames.max()]

plt.figure(figsize=[5,3])
plt.plot((frames**2).max(axis=1).max(axis=1),label="max(Im)")
#plt.plot(np.cumsum(frames.max(axis=1).max(axis=1))/(1+np.arange(frames.shape[0])))
plt.ylabel("Strehl")
plt.xlabel("Iteration")
plt.savefig("strehlvstime.png")

fig,ax = plt.subplots(1,2,figsize=[8,4])
ax[0].imshow(frames[0])
ax[0].images[0].set_clim(clim)
ax[0].set_title("OIWFS Image - Frame 1")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(frames[0])
ax[1].images[0].set_clim(clim)
ax[1].set_title("20-frame Avg")
ax[1].set_xticks([])
ax[1].set_yticks([])

def update(frame_num):
    ax[0].images[0].set_data(frames[frame_num])
    ax[0].set_title(f"OIWFS Image - Frame {frame_num+1:d}")
    ax[1].images[0].set_data(frames[frame_num-20:frame_num+1].mean(axis=0))
    return ax

anim = FuncAnimation(fig,update,range(len(frames)),interval=0.1, repeat=False)

anim.save("closed_loop_gym.gif")