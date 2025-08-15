#!/usr/env python

import numpy as np
import gymnasium as gym
import segment_phasing_fp_env  # noqa: F401
from segment_phasing_fp_env import psf
from tqdm import tqdm
from pripy.algos import MHE

# number of previous states to consider in MHE
NBUFFER: int = 3
# filter gain, can be up to 1.0 and still stable
GAIN: float = 0.2

if __name__ == "__main__":
    # create a model to be used for calibrating controller
    model = psf.PSF(ideal=True)
    model.state *= 0.0
    model.command *= 0.0
    # build controller
    ctrl = MHE.from_model(model, nbuffer=NBUFFER)

    # make environment
    env = gym.make("SegmentPhasingFP-v0")

    # extract environment parameters
    observation, info = env.reset(seed=42)
    if env.action_space.shape is not None:
        nactu = env.action_space.shape[0]
    else:
        raise ValueError("Environment has no action space")
    if env.observation_space.shape is not None:
        nmeas = np.prod(env.observation_space.shape)
    else:
        raise ValueError("Enviornment has no measurement space")
    nmodes = model.state.shape[0]

    # initialise variables
    frames = []
    strehl = []
    x0 = np.zeros([NBUFFER, nmodes]).flatten()
    x_dm = np.zeros([NBUFFER, nmodes]).flatten()
    yd = np.zeros([NBUFFER, nmeas])
    com = np.zeros(nmodes)
    old_com = com.copy()

    # run control loop
    for it in (pbar := tqdm(range(500), disable=False)):
        # compute action from command. Actions are cumulative, but the command
        # is calculated as the negated estimate of the state of the system (in
        # pseudo-open-loop), therefore, the action is the "incremental"
        # command:
        action = com - old_com

        # grab a frame
        frame, reward, terminated, truncated, info = env.step(action)

        # update telemetry buffers/logs
        yd[:-1, ...] = yd[1:, ...]
        yd[-1, ...] = frame.flatten()
        x_dm[:-nmodes] = x_dm[nmodes:]
        x_dm[-nmodes:] = com
        old_com = com.copy()
        frames.append(frame)
        strehl.append(info["se_strehl"])
        pbar.set_description(
            f"reward: {reward:0.3e}, se sr: {info['se_strehl']:0.3f}"
        )

        # if we are ready, run an iteration of the controller
        if it > NBUFFER and it > 100:
            # compute the estimated state
            x_hat = ctrl.get_estimate(x0, x_dm, yd)
            # update the command based on the estimate
            com = (1 - GAIN) * com - GAIN * x_hat
            # update the initial estimate for next iteration
            x0[:-nmodes] = x0[nmodes:]
            x0[-nmodes:] = x_hat

    # save frames for review
    np.save("frames.npy", np.array(frames))
    np.save("strehl.npy", np.array(strehl))
