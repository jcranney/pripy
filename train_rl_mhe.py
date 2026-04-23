import gymnasium as gym
import numpy as np
from gymnasium import spaces
try:
    from sbx import PPO               # JAX-based PPO (preferred, much faster)
    _USING_JAX_PPO = True
except ImportError:
    from stable_baselines3 import PPO  # PyTorch fallback
    _USING_JAX_PPO = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

import segment_phasing_fp_env
from segment_phasing_fp_env import psf_autodiff as psf
from pripy.algos import MHE


# ===== Hyperparameters =====
NBUFFER         = 3
N_STACK         = 3      # PSF frames stacked in observation (addresses single-frame limitation)
GAIN            = 0.2
EPISODE_LEN     = 200    # steps per training episode
ROLLOUT_LEN     = 500    # steps for evaluation rollout
RL_MAX_ACTION   = 0.01   # RL action space directly bounded to [-0.01, 0.01]
TRAIN_TIMESTEPS = 200_000

ACTION_PENALTY_COEF = 0.05
STREHL_DROP_PENALTY = 3.0

# ----- Primary Strehl-term shaping -----------------------------------------
# Primary reward signal is a sigmoid-mapped short-exposure Strehl ratio:
#     sigma(beta * (strehl - tau))
# which concentrates the learning gradient in the collapse-regime Strehl
# range [0.3, 0.7] and saturates near 1 above tau, preventing marginal
# steady-state gains from dominating the optimisation. Setting
# USE_SIGMOID_REWARD = False falls back to the raw-Strehl primary signal
# used as the ablation variant in the reward-shaping study.
USE_SIGMOID_REWARD = True
SIGMOID_BETA       = 8.0
SIGMOID_TAU        = 0.5


def _map_strehl(strehl: float) -> float:
    """Apply the primary-signal mapping to an instantaneous Strehl value."""
    if USE_SIGMOID_REWARD:
        return float(1.0 / (1.0 + np.exp(-SIGMOID_BETA * (strehl - SIGMOID_TAU))))
    return float(strehl)


def _strehl_reward(strehl: float, prev_strehl: float, action_norm: float) -> float:
    """
    Reward = primary signal                        (sigmoid-mapped by default)
           - large penalty when Strehl drops       (promotes stability)
           - small penalty for large RL actions    (suppresses jitter)
    The drop penalty is defined on the raw Strehl difference, independent of
    the primary-signal mapping, so the penalty scale is invariant to the
    choice of USE_SIGMOID_REWARD.
    """
    primary        = _map_strehl(strehl)
    drop_penalty   = STREHL_DROP_PENALTY * max(0.0, prev_strehl - strehl)
    action_penalty = ACTION_PENALTY_COEF * action_norm
    return float(primary - drop_penalty - action_penalty)


class RLMHEEnv(gym.Env):
    """
    RL environment wrapping MHE control.

    Observation (1218-dim):
        - N_STACK=3 consecutive normalised PSF frames  (3 × 400 = 1200 dims)
        - N_STACK=3 corresponding MHE commands         (3 × 6   = 18 dims)
      The RL agent receives both temporal context (stacked frames) and the
      full history of MHE control signals associated with those frames.

    Action (6-dim, bounded [-0.01, 0.01]):
        Residual correction added on top of the MHE increment.
        Action space is directly at the target scale (no external ε factor).
    """
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = EPISODE_LEN):
        super().__init__()
        self.env = gym.make("SegmentPhasingFP-v0")
        self.env.unwrapped.max_steps = max_steps

        self.model = psf.PSFAutoDiff(ideal=True)
        self.model.state   *= 0.0
        self.model.command *= 0.0
        self.ctrl = MHE.from_model(self.model, nbuffer=NBUFFER, use_jax=True)

        self.nmodes = self.model.state.shape[0]   # 6
        self.nmeas  = 20 * 20                      # 400

        # obs = [frame_{t-2}, frame_{t-1}, frame_t, com_{t-2}, com_{t-1}, com_t]
        psf_dim = N_STACK * self.nmeas     # 1200
        com_dim = N_STACK * self.nmodes    # 18
        self.observation_space = spaces.Box(
            low  = np.concatenate([np.zeros(psf_dim, dtype=np.float32),
                                   np.full(com_dim, -2.0, dtype=np.float32)]),
            high = np.concatenate([np.ones(psf_dim, dtype=np.float32),
                                   np.full(com_dim,  2.0, dtype=np.float32)]),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-RL_MAX_ACTION, high=RL_MAX_ACTION,
            shape=(self.nmodes,), dtype=np.float32,
        )

        self.current_step = 0
        self.frame_buffer = None   # (N_STACK * nmeas,)
        self.com_buffer   = None   # (N_STACK * nmodes,)
        self.x0 = self.x_dm = self.yd = self.com = self.old_com = None
        self.last_raw_obs = None
        self.prev_strehl  = 0.0

    # ------------------------------------------------------------------
    def _preprocess_frame(self, obs) -> np.ndarray:
        return (obs.astype(np.float32) / 65_535.0).flatten()

    def _init_controller_buffers(self):
        self.x0          = np.zeros([NBUFFER, self.nmodes], dtype=np.float64).flatten()
        self.x_dm        = np.zeros([NBUFFER, self.nmodes], dtype=np.float64).flatten()
        self.yd          = np.zeros([NBUFFER, self.nmeas],  dtype=np.float32)
        self.com         = np.zeros(self.nmodes, dtype=np.float32)
        self.old_com     = self.com.copy()
        self.frame_buffer = np.zeros(N_STACK * self.nmeas,  dtype=np.float32)
        self.com_buffer   = np.zeros(N_STACK * self.nmodes, dtype=np.float32)

    def _build_obs(self, normalized_frame: np.ndarray) -> np.ndarray:
        """Shift both buffers left, insert new entries, concatenate."""
        self.frame_buffer[:-self.nmeas]  = self.frame_buffer[self.nmeas:]
        self.frame_buffer[-self.nmeas:]  = normalized_frame
        self.com_buffer[:-self.nmodes]   = self.com_buffer[self.nmodes:]
        self.com_buffer[-self.nmodes:]   = self.com.astype(np.float32)
        return np.concatenate([self.frame_buffer, self.com_buffer])

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.current_step = 0
        self.prev_strehl  = 0.0
        self._init_controller_buffers()
        self.last_raw_obs = obs
        self.yd[-1, ...] = obs.flatten()

        normalized = self._preprocess_frame(obs)
        self.frame_buffer[:] = np.tile(normalized, N_STACK)
        # com_buffer already zero-initialised (matching com = zeros)
        init_obs = np.concatenate([self.frame_buffer, self.com_buffer])
        return init_obs, info

    def step(self, action):
        self.current_step += 1

        # Shift MHE buffers
        self.yd[:-1,  ...] = self.yd[1:,  ...]
        self.yd[-1,   ...] = self.last_raw_obs.flatten()
        self.x_dm[:-self.nmodes] = self.x_dm[self.nmodes:]
        self.x_dm[-self.nmodes:] = self.com
        self.old_com = self.com.copy()

        # Run MHE before applying action
        if self.current_step >= NBUFFER:
            x_hat = self.ctrl.get_estimate(self.x0, self.x_dm, self.yd)
            self.com = (1.0 - GAIN) * self.com - GAIN * x_hat
            self.x0[:-self.nmodes] = self.x0[self.nmodes:]
            self.x0[-self.nmodes:] = x_hat

        mhe_increment = self.com - self.old_com
        rl_delta      = np.clip(action, -RL_MAX_ACTION, RL_MAX_ACTION)

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

        normalized = self._preprocess_frame(obs)
        return self._build_obs(normalized), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Run pure MHE (no RL) for comparison
# Buffer handling mirrors RLMHEEnv.step exactly so that eps=0 and MHE-only
# produce identical results with the same seed.
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
    yd   = np.zeros([NBUFFER, nmeas],  dtype=np.float32)
    com  = np.zeros(nmodes, dtype=np.float32)

    last_raw_obs = obs
    yd[-1, ...] = obs.flatten()

    frames     = []
    se_strehls = []
    le_strehls = []

    for step in tqdm(range(1, ROLLOUT_LEN + 1), desc="MHE-only rollout"):
        yd[:-1, ...] = yd[1:, ...]
        yd[-1,  ...] = last_raw_obs.flatten()
        x_dm[:-nmodes] = x_dm[nmodes:]
        x_dm[-nmodes:] = com
        old_com = com.copy()

        if step >= NBUFFER:
            x_hat  = ctrl.get_estimate(x0, x_dm, yd)
            com    = (1.0 - GAIN) * com - GAIN * x_hat
            x0[:-nmodes] = x0[nmodes:]
            x0[-nmodes:] = x_hat

        mhe_increment = com - old_com

        obs, _, _, truncated, info = base_env.step(mhe_increment)
        last_raw_obs = obs

        se_strehl = float(np.clip(info.get("se_strehl", 0.0), 0.0, 1.0))
        le_strehl = float(np.clip(info.get("le_strehl", 0.0), 0.0, 1.0))
        frames.append((obs.astype(np.float32) / 65_535.0).reshape(20, 20))
        se_strehls.append(se_strehl)
        le_strehls.append(le_strehl)

        if truncated:
            break

    return frames, se_strehls, le_strehls


# ---------------------------------------------------------------------------
# Run RL+MHE with zero RL actions (epsilon = 0 baseline)
# ---------------------------------------------------------------------------
def run_eps0(seed: int = 0) -> tuple[list, list, list]:
    """Run RLMHEEnv with zero RL actions — MHE-only through the same wrapper.
    Returns (frames, se_strehls, le_strehls).
    """
    env = RLMHEEnv(max_steps=ROLLOUT_LEN)
    obs, _ = env.reset(seed=seed)

    frames     = []
    se_strehls = []
    le_strehls = []

    for _ in tqdm(range(ROLLOUT_LEN), desc="eps=0 rollout"):
        zero_action = np.zeros(env.nmodes, dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(zero_action)
        # Latest frame = last 400 of the frame section (before com_buffer)
        frame_end = N_STACK * env.nmeas
        frames.append(obs[frame_end - env.nmeas : frame_end].reshape(20, 20).copy())
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
# Save comparison plot: MHE-only vs eps=0 vs RL+MHE
# ---------------------------------------------------------------------------
def save_comparison(
    se_mhe:  list, le_mhe:  list,
    se_eps0: list, le_eps0: list,
    se_rl:   list, le_rl:   list,
    out: str = "comparison.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    datasets = [
        (se_mhe,  le_mhe,  "#888888", "MHE only",                          "--", 2.5),
        (se_eps0, le_eps0, "#e08800", "RL+MHE  (ε=0)",                     "-",  1.5),
        (se_rl,   le_rl,   "#2244cc", f"RL+MHE  (action∈[±{RL_MAX_ACTION}])", "-", 1.5),
    ]

    titles = ["Short-Exposure (instantaneous) Strehl", "Long-Exposure Strehl"]

    for col, (ax, title) in enumerate(zip(axes, titles)):
        for se_data, le_data, color, label, ls, lw in datasets:
            data = se_data if col == 0 else le_data
            xs   = np.arange(1, len(data) + 1)
            ax.plot(xs, data, color=color, linewidth=lw, linestyle=ls, label=label)

        ax.set_xlim(0, ROLLOUT_LEN)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("#iteration", fontsize=12)
        ax.set_ylabel("Strehl ratio", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, color="#e0e0e0")
        ax.set_facecolor("#eef0f8")

    plt.suptitle(f"MHE only  vs  RL+MHE (ε=0)  vs  RL+MHE  "
                 f"[N_STACK={N_STACK}, action∈[±{RL_MAX_ACTION}]]", fontsize=12)
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
        ("MHE only",                   se_mhe,  le_mhe),
        ("RL+MHE  ε=0",                se_eps0, le_eps0),
        (f"RL+MHE  ε=[±{RL_MAX_ACTION}]", se_rl, le_rl),
    ]

    print("\n" + "="*72)
    print(f"{'Metric':<26} {'SE mean(last100)':<18} {'LE mean(last100)':<18} {'Steps→SE>0.9'}")
    print("-"*72)
    for label, se, le in rows:
        s2t = steps_to_threshold(se, 0.9)
        print(f"{label:<26} {mean_last(se):<18.4f} {mean_last(le):<18.4f} {str(s2t)}")
    print("="*72 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # ── 1. Train (skipped if ppo_rl_mhe.zip already exists) ───────────────
    print(f"PPO backend: {'SBX (JAX)' if _USING_JAX_PPO else 'stable-baselines3 (PyTorch)'}")

    if os.path.exists("ppo_rl_mhe.zip"):
        print("Found existing ppo_rl_mhe.zip — skipping training, loading model …")
        model = PPO.load("ppo_rl_mhe")
    else:
        train_env = RLMHEEnv()

        ppo_kwargs = dict(
            policy="MlpPolicy",
            env=train_env,
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
        )
        if not _USING_JAX_PPO:
            ppo_kwargs["device"] = "cpu"

        model = PPO(**ppo_kwargs)
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
        frame_end = N_STACK * rl_env.nmeas
        rl_frames.append(obs[frame_end - rl_env.nmeas : frame_end].reshape(20, 20).copy())
        rl_se.append(float(info.get("se_strehl", 0.0)))
        rl_le.append(float(np.clip(info.get("le_strehl", 0.0), 0.0, 1.0)))
        if terminated or truncated:
            break

    # ── 3. eps=0 rollout ──────────────────────────────────────────────────
    print("\nRolling out eps=0 episode …")
    _, eps0_se, eps0_le = run_eps0(seed=0)

    # ── 4. MHE-only rollout ───────────────────────────────────────────────
    print("\nRolling out MHE-only episode …")
    mhe_frames, mhe_se, mhe_le = run_mhe_only(seed=0)

    # ── 5. Summary table ──────────────────────────────────────────────────
    print_summary(mhe_se, mhe_le, eps0_se, eps0_le, rl_se, rl_le)

    # ── 6. Save GIF ───────────────────────────────────────────────────────
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
