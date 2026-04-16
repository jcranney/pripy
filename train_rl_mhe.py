import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

import segment_phasing_fp_env
from segment_phasing_fp_env import psf_autodiff as psf
from pripy.algos import MHE


# ===== Hyperparameters =====
NBUFFER          = 3
GAIN             = 0.2
EPISODE_LEN      = 200   # used during PPO training (keep short for speed)
ROLLOUT_LEN      = 500   # used for final GIF/comparison
DELTA_SCALE      = 1.0       # scaling removed: action_space is now [-0.01, 0.01] directly
TRAIN_TIMESTEPS  = 200_000

GATE_START = 0    # RL starts from step 0
GATE_END   = 60   # RL reaches full authority by step 60

ACTION_PENALTY_COEF  = 0.05
STREHL_DROP_PENALTY  = 3.0


def _strehl_reward(strehl: float, prev_strehl: float, action_norm: float) -> float:
    """
    Reward = sigmoid(Strehl)
           - large penalty when Strehl drops
           - small penalty for large RL actions
    """
    k = 10.0
    reward_strehl  = 1.0 / (1.0 + np.exp(-k * (strehl - 0.5)))
    drop_penalty   = STREHL_DROP_PENALTY * max(0.0, prev_strehl - strehl)
    action_penalty = ACTION_PENALTY_COEF * action_norm
    return float(reward_strehl - drop_penalty - action_penalty)


class RLMHEEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = EPISODE_LEN):
        super().__init__()
        self.env = gym.make("SegmentPhasingFP-v0")
        self.env.unwrapped.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(400,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-0.01, high=0.01, shape=(6,), dtype=np.float32,
        )

        self.model = psf.PSFAutoDiff(ideal=True)
        self.model.state   *= 0.0
        self.model.command *= 0.0
        self.ctrl = MHE.from_model(self.model, nbuffer=NBUFFER, use_jax=True)

        self.nmodes = self.model.state.shape[0]
        self.nmeas  = 20 * 20
        self.current_step = 0

        self.x0 = self.x_dm = self.yd = self.com = self.old_com = None
        self.last_raw_obs = None
        self.prev_strehl  = 0.0

    def _preprocess_obs(self, obs):
        return (obs.astype(np.float32) / 65_535.0).flatten()

    def _init_controller_buffers(self):
        self.x0      = np.zeros([NBUFFER, self.nmodes], dtype=np.float64).flatten()
        self.x_dm    = np.zeros([NBUFFER, self.nmodes], dtype=np.float64).flatten()
        self.yd      = np.zeros([NBUFFER, self.nmeas],  dtype=np.float32)
        self.com     = np.zeros(self.nmodes, dtype=np.float32)
        self.old_com = self.com.copy()

    def _rl_gate_weight(self):
        if self.current_step <= GATE_START:
            return 0.0
        if self.current_step >= GATE_END:
            return 1.0
        return (self.current_step - GATE_START) / (GATE_END - GATE_START)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.current_step = 0
        self.prev_strehl  = 0.0
        self._init_controller_buffers()
        self.last_raw_obs = obs
        self.yd[-1, ...] = obs.flatten()
        return self._preprocess_obs(obs), info

    def step(self, action):
        self.current_step += 1

        self.yd[:-1,  ...] = self.yd[1:,  ...]
        self.yd[-1,   ...] = self.last_raw_obs.flatten()
        self.x_dm[:-self.nmodes] = self.x_dm[self.nmodes:]
        self.x_dm[-self.nmodes:] = self.com
        self.old_com = self.com.copy()

        if self.current_step >= NBUFFER:
            x_hat = self.ctrl.get_estimate(self.x0, self.x_dm, self.yd)
            self.com = (1.0 - GAIN) * self.com - GAIN * x_hat
            self.x0[:-self.nmodes] = self.x0[self.nmodes:]
            self.x0[-self.nmodes:] = x_hat
        else:
            x_hat = np.zeros(self.nmodes, dtype=np.float32)

        mhe_increment = self.com - self.old_com
        gate_weight   = self._rl_gate_weight()
        rl_delta      = gate_weight * np.clip(action, -1.0, 1.0) * DELTA_SCALE

        final_action = np.clip(
            mhe_increment + rl_delta,
            self.env.action_space.low,
            self.env.action_space.high,
        )

        obs, _, _, _, info = self.env.step(final_action)
        self.last_raw_obs = obs

        raw_strehl     = float(info.get("se_strehl", 0.0))
        current_strehl = float(np.clip(raw_strehl, 0.0, 1.0))
        reward         = _strehl_reward(current_strehl, self.prev_strehl,
                                        float(np.linalg.norm(rl_delta)))
        self.prev_strehl = current_strehl

        terminated = False
        truncated  = self.current_step >= self.env.unwrapped.max_steps

        info = dict(info)
        info["se_strehl"] = current_strehl

        return self._preprocess_obs(obs), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Run pure MHE (no RL) for comparison
# FIX: MHE now runs BEFORE env.step so the increment is actually applied.
#      Previously action = com - old_com was always 0 (bug).
# ---------------------------------------------------------------------------
def run_mhe_only(seed: int = 0) -> tuple[list, list, list]:
    """Run the MHE controller without any RL policy.
    Returns (frames, se_strehls, le_strehls).
    """
    base_env = gym.make("SegmentPhasingFP-v0")
    base_env.unwrapped.max_steps = ROLLOUT_LEN
    mhe_model = psf.PSFAutoDiff(ideal=True)
    mhe_model.state   *= 0.0
    mhe_model.command *= 0.0
    ctrl = MHE.from_model(mhe_model, nbuffer=NBUFFER, use_jax=True)

    nmodes = mhe_model.state.shape[0]
    nmeas  = 20 * 20

    obs, _ = base_env.reset(seed=seed)

    x0   = np.zeros([NBUFFER, nmodes], dtype=np.float64).flatten()
    x_dm = np.zeros([NBUFFER, nmodes], dtype=np.float64).flatten()
    yd   = np.zeros([NBUFFER, nmeas],  dtype=np.float64)
    com  = np.zeros(nmodes, dtype=np.float32)

    frames     = []
    se_strehls = []
    le_strehls = []

    for it in tqdm(range(ROLLOUT_LEN), desc="MHE-only rollout"):
        old_com = com.copy()

        # Shift observation and command buffers
        yd[:-1, ...] = yd[1:, ...]
        yd[-1,  ...] = obs.flatten()
        x_dm[:-nmodes] = x_dm[nmodes:]
        x_dm[-nmodes:] = com

        # Run MHE BEFORE env.step so the computed increment is applied this step
        if it >= NBUFFER:
            x_hat  = ctrl.get_estimate(x0, x_dm, yd)
            com    = (1.0 - GAIN) * com - GAIN * x_hat
            x0[:-nmodes] = x0[nmodes:]
            x0[-nmodes:] = x_hat

        mhe_increment = com - old_com   # actual non-zero correction

        obs, _, _, truncated, info = base_env.step(mhe_increment)
        # Ignore terminated (Strehl<0.1 early-stop) to match RLMHEEnv behaviour,
        # which hardcodes terminated=False and always runs the full rollout.

        se_strehl = float(np.clip(info.get("se_strehl", 0.0), 0.0, 1.0))
        le_strehl = float(np.clip(info.get("le_strehl", 0.0), 0.0, 1.0))
        frames.append((obs.astype(np.float32) / 65_535.0).reshape(20, 20))
        se_strehls.append(se_strehl)
        le_strehls.append(le_strehl)

        if truncated:
            break

    return frames, se_strehls, le_strehls


# ---------------------------------------------------------------------------
# Run RL+MHE with zero RL actions (epsilon = 0 baseline through the RL wrapper)
# ---------------------------------------------------------------------------
def run_eps0(seed: int = 0) -> tuple[list, list, list]:
    """Run the RLMHEEnv but always pass zero RL actions.
    This shows what MHE contributes through the same wrapper as the trained agent,
    providing a fair epsilon=0 baseline.
    Returns (frames, se_strehls, le_strehls).
    """
    env = RLMHEEnv(max_steps=ROLLOUT_LEN)
    obs, _ = env.reset(seed=seed)

    frames     = []
    se_strehls = []
    le_strehls = []

    for _ in tqdm(range(ROLLOUT_LEN), desc="eps=0 rollout"):
        zero_action = np.zeros(6, dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(zero_action)
        frames.append(obs.reshape(20, 20).copy())
        se_strehls.append(float(info.get("se_strehl", 0.0)))
        le_strehls.append(float(np.clip(info.get("le_strehl", 0.0), 0.0, 1.0)))
        if terminated or truncated:
            break

    return frames, se_strehls, le_strehls


# ---------------------------------------------------------------------------
# Save GIF
# ---------------------------------------------------------------------------
def save_gif(
    frames: list,
    strehls: list,
    fps: int = 20,
    out: str = "simul.gif",
) -> None:
    n = len(frames)
    print(f"Building GIF ({n} frames @ {fps} fps) …")

    fig, (ax_psf, ax_sr) = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")

    im = ax_psf.imshow(
        frames[0], cmap="hot", vmin=0.0, vmax=1.0,
        interpolation="nearest", origin="upper",
    )
    ax_psf.set_xticks([])
    ax_psf.set_yticks([])
    title = ax_psf.set_title(
        f"frame 001/{n:03d}, strehl: {strehls[0]:.3f}", fontsize=11,
    )

    ax_sr.set_xlim(0, n)
    ax_sr.set_ylim(0.0, 1.0)
    ax_sr.set_xlabel("#iteration", fontsize=11)
    ax_sr.set_ylabel("Strehl ratio", fontsize=11)
    ax_sr.grid(True, color="#e0e0e0")
    ax_sr.set_facecolor("#eef0f8")
    (line,) = ax_sr.plot([], [], color="#2244cc", linewidth=1.5)

    plt.tight_layout()

    def _update(i):
        im.set_data(frames[i])
        title.set_text(f"frame {i+1:03d}/{n:03d}, strehl: {strehls[i]:.3f}")
        line.set_data(np.arange(1, i + 2), strehls[:i + 1])
        return im, title, line

    anim = FuncAnimation(fig, _update, frames=n, interval=1000 / fps, blit=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Save comparison plot: MHE-only vs eps=0 vs RL+MHE  (SE and LE Strehl)
# ---------------------------------------------------------------------------
def save_comparison(
    se_mhe:  list, le_mhe:  list,
    se_eps0: list, le_eps0: list,
    se_rl:   list, le_rl:   list,
    out: str = "comparison.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    datasets = [
        (se_mhe,  le_mhe,  "#888888", "MHE only"),
        (se_eps0, le_eps0, "#e08800", "RL+MHE  (ε=0)"),
        (se_rl,   le_rl,   "#2244cc", "RL+MHE  (action∈[-0.01,0.01])"),
    ]

    titles = ["Short-Exposure (instantaneous) Strehl", "Long-Exposure Strehl"]

    for col, (ax, title) in enumerate(zip(axes, titles)):
        for se_data, le_data, color, label in datasets:
            data = se_data if col == 0 else le_data
            xs   = np.arange(1, len(data) + 1)
            ax.plot(xs, data, color=color, linewidth=1.5, label=label)

        ax.axvline(GATE_START, color="orange", linestyle="--", linewidth=1.0,
                   label=f"RL gate start ({GATE_START})")
        ax.axvline(GATE_END,   color="red",    linestyle="--", linewidth=1.0,
                   label=f"RL gate end ({GATE_END})")
        ax.set_xlim(0, ROLLOUT_LEN)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("#iteration", fontsize=12)
        ax.set_ylabel("Strehl ratio", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, color="#e0e0e0")
        ax.set_facecolor("#eef0f8")

    plt.suptitle("MHE only  vs  RL+MHE (ε=0)  vs  RL+MHE", fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
def print_summary(
    se_mhe:  list, le_mhe:  list,
    se_eps0: list, le_eps0: list,
    se_rl:   list, le_rl:   list,
) -> None:
    def mean_last(data, n=100):
        return float(np.mean(data[-n:])) if len(data) >= n else float(np.mean(data))

    def steps_to_threshold(data, thresh=0.9):
        for i, v in enumerate(data):
            if v >= thresh:
                return i + 1
        return None

    rows = [
        ("MHE only",        se_mhe,  le_mhe),
        ("RL+MHE  ε=0",     se_eps0, le_eps0),
        (f"RL+MHE  ε={DELTA_SCALE}", se_rl, le_rl),
    ]

    print("\n" + "="*70)
    print(f"{'Metric':<22} {'SE mean(last 100)':<20} {'LE mean(last 100)':<20} {'Steps to SE>0.9'}")
    print("-"*70)
    for label, se, le in rows:
        s2t = steps_to_threshold(se, 0.9)
        print(f"{label:<22} {mean_last(se):<20.4f} {mean_last(le):<20.4f} {str(s2t):<15}")
    print("="*70 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # ── 1. Train (skipped if ppo_rl_mhe.zip already exists) ───────────────
    if os.path.exists("ppo_rl_mhe.zip"):
        print("Found existing ppo_rl_mhe.zip — skipping training, loading model …")
        model = PPO.load("ppo_rl_mhe")
    else:
        train_env = RLMHEEnv()
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cpu",
        )
        print(f"Training PPO for {TRAIN_TIMESTEPS:,} timesteps …")
        model.learn(total_timesteps=TRAIN_TIMESTEPS)
        model.save("ppo_rl_mhe")
        print("Model saved → ppo_rl_mhe.zip\n")

    # ── 2. RL+MHE rollout ─────────────────────────────────────────────────
    print("Rolling out RL+MHE episode …")
    rl_env = RLMHEEnv(max_steps=ROLLOUT_LEN)
    obs, _ = rl_env.reset(seed=0)
    rl_frames, rl_se, rl_le = [], [], []
    for _ in tqdm(range(ROLLOUT_LEN), desc="RL+MHE rollout"):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = rl_env.step(action)
        rl_frames.append(obs.reshape(20, 20).copy())
        rl_se.append(float(info.get("se_strehl", 0.0)))
        rl_le.append(float(np.clip(info.get("le_strehl", 0.0), 0.0, 1.0)))
        if terminated or truncated:
            break

    # ── 3. eps=0 rollout (MHE through RL wrapper, zero RL actions) ────────
    print("\nRolling out eps=0 episode …")
    _, eps0_se, eps0_le = run_eps0(seed=0)

    # ── 4. MHE-only rollout (fixed baseline) ──────────────────────────────
    print("\nRolling out MHE-only episode …")
    mhe_frames, mhe_se, mhe_le = run_mhe_only(seed=0)

    # ── 5. Print summary table ─────────────────────────────────────────────
    print_summary(mhe_se, mhe_le, eps0_se, eps0_le, rl_se, rl_le)

    # ── 6. Save GIF (RL+MHE) ──────────────────────────────────────────────
    print()
    save_gif(rl_frames, rl_se, fps=20, out="simul.gif")

    # ── 7. Save comparison plot ───────────────────────────────────────────
    save_comparison(
        mhe_se,  mhe_le,
        eps0_se, eps0_le,
        rl_se,   rl_le,
        out="comparison.png",
    )

    print("\nDone! Generated files:")
    print("  simul.gif      ← animated PSF + SE Strehl (RL+MHE)")
    print("  comparison.png ← SE & LE Strehl: MHE only | eps=0 | RL+MHE")
