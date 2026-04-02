import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

import segment_phasing_fp_env
from segment_phasing_fp_env import psf_autodiff as psf
from pripy.algos import MHE


# ===== Hyperparameters =====
NBUFFER = 3
GAIN = 0.2
EPISODE_LEN = 200
DELTA_SCALE = 0.01         # Maximum RL residual magnitude (in actuator units)
TRAIN_TIMESTEPS = 20_000

# --- Action gating schedule ---
# Before GATE_START: RL is fully disabled (pure MHE).
# Between GATE_START and GATE_END: RL weight ramps linearly from 0 → 1.
# After GATE_END: RL operates at full DELTA_SCALE.
GATE_START = 40            # step at which RL begins to contribute
GATE_END   = 120           # step at which RL reaches full authority

# --- Reward shaping ---
ACTION_PENALTY_COEF = 0.05  # penalises large RL deltas to suppress jitter


# ---------------------------------------------------------------------------
# Helper: sigmoid-based Strehl reward that is smooth and bounded in [0, 1]
# ---------------------------------------------------------------------------
def _strehl_reward(strehl: float, action_norm: float) -> float:
    """
    Continuous, shaped reward:
      • Maps Strehl ∈ [0, 1] to a smooth reward via a shifted sigmoid so that
        rewards grow steeply around a "good" operating regime (≥ 0.5 Strehl)
        instead of a hard threshold.
      • Subtracts a small penalty proportional to the L2 norm of the RL action
        to discourage unnecessarily large corrections.

    Returns a scalar in roughly [-0.1, 1.0].
    """
    # Sigmoid centred at Strehl = 0.5, steepness k = 10
    # reward_strehl ∈ (0, 1) for any valid Strehl
    k = 10.0
    reward_strehl = 1.0 / (1.0 + np.exp(-k * (strehl - 0.5)))

    # Action-magnitude penalty (L2 norm already normalised to [0, 1] by
    # DELTA_SCALE in the step() method, so action_norm ≤ sqrt(6) ≈ 2.45)
    penalty = ACTION_PENALTY_COEF * action_norm

    return float(reward_strehl - penalty)


class RLMHEEnv(gym.Env):
    """
    RL-augmented MHE controller environment.

    Key design decisions
    --------------------
    1. **Strehl clamping** – raw Strehl from the env is clipped to [0, 1]
       before any reward calculation or logging, eliminating spurious >1 values.

    2. **Smooth RL gating** – instead of a hard cut-over at step 80, the RL
       authority ramps linearly from 0 → 1 over [GATE_START, GATE_END].  This
       prevents the abrupt disturbance that was causing post-step-40 collapses.

    3. **Shaped reward** – a sigmoid reward + action-magnitude penalty replaces
       the discontinuous threshold reward, giving the policy a dense gradient
       throughout the Strehl range.

    4. **Consistent observation scaling** – observations are always divided by
       the sensor bit-depth maximum (65 535) and flattened to float32, matching
       the declared Box(0, 1, shape=(400,)) observation space exactly.
       Actions are kept in [-1, 1] (Box convention) and scaled by DELTA_SCALE
       inside step(), so the network always operates in a normalised space.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Base phasing environment
        self.env = gym.make("SegmentPhasingFP-v0")

        # RL observation: normalised flattened 20×20 focal-plane image
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(400,),
            dtype=np.float32,
        )

        # RL action: residual correction in normalised [-1, 1] per mode
        # Physical scale applied inside step() via DELTA_SCALE
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # Build MHE controller model
        self.model = psf.PSFAutoDiff(ideal=True)
        self.model.state   *= 0.0
        self.model.command *= 0.0
        self.ctrl = MHE.from_model(self.model, nbuffer=NBUFFER, use_jax=True)

        self.nmodes = self.model.state.shape[0]
        self.nmeas  = 20 * 20

        self.current_step = 0

        # Controller state variables (initialised in reset)
        self.x0        = None
        self.x_dm      = None
        self.yd        = None
        self.com       = None
        self.old_com   = None
        self.last_raw_obs = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalise raw 16-bit sensor frame to float32 ∈ [0, 1]."""
        return (obs.astype(np.float32) / 65_535.0).flatten()

    def _init_controller_buffers(self):
        """Zero-initialise all MHE history arrays."""
        self.x0      = np.zeros([NBUFFER, self.nmodes], dtype=np.float64).flatten()
        self.x_dm    = np.zeros([NBUFFER, self.nmodes], dtype=np.float64).flatten()
        self.yd      = np.zeros([NBUFFER, self.nmeas],  dtype=np.float32)
        self.com     = np.zeros(self.nmodes, dtype=np.float32)
        self.old_com = self.com.copy()

    def _rl_gate_weight(self) -> float:
        """
        Linear ramp from 0 → 1 over [GATE_START, GATE_END].
        Returns 0.0 before ramp, 1.0 after ramp.
        """
        if self.current_step <= GATE_START:
            return 0.0
        if self.current_step >= GATE_END:
            return 1.0
        return (self.current_step - GATE_START) / (GATE_END - GATE_START)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.current_step = 0
        self._init_controller_buffers()

        self.last_raw_obs = obs
        self.yd[-1, ...] = obs.flatten()

        return self._preprocess_obs(obs), info

    def step(self, action: np.ndarray):
        self.current_step += 1

        # =========================================================
        # 1) Shift MHE history buffers
        # =========================================================
        self.yd[:-1,  ...] = self.yd[1:,  ...]
        self.yd[-1,   ...] = self.last_raw_obs.flatten()

        self.x_dm[:-self.nmodes] = self.x_dm[self.nmodes:]
        self.x_dm[-self.nmodes:] = self.com

        self.old_com = self.com.copy()

        # =========================================================
        # 2) MHE estimate (once enough history is available)
        # =========================================================
        if self.current_step >= NBUFFER:
            x_hat = self.ctrl.get_estimate(self.x0, self.x_dm, self.yd)
            self.com = (1.0 - GAIN) * self.com - GAIN * x_hat

            self.x0[:-self.nmodes] = self.x0[self.nmodes:]
            self.x0[-self.nmodes:] = x_hat
        else:
            x_hat = np.zeros(self.nmodes, dtype=np.float32)

        mhe_increment = self.com - self.old_com

        # =========================================================
        # 3) RL residual with smooth gating
        #    action ∈ [-1, 1]  →  rl_delta ∈ [-DELTA_SCALE, DELTA_SCALE]
        #    gate_weight ramps 0 → 1 so the residual phases in gradually.
        # =========================================================
        gate_weight = self._rl_gate_weight()
        # Clamp network output strictly to [-1, 1] before scaling
        action_clamped = np.clip(action, -1.0, 1.0)
        rl_delta = gate_weight * action_clamped * DELTA_SCALE

        final_action = np.clip(
            mhe_increment + rl_delta,
            self.env.action_space.low,
            self.env.action_space.high,
        )

        # =========================================================
        # 4) Apply action to the phasing environment
        # =========================================================
        obs, _, terminated, truncated, info = self.env.step(final_action)
        self.last_raw_obs = obs

        # =========================================================
        # 5) Physics-correct Strehl: clamp to [0, 1]
        #    Raw env values > 1 (e.g. 1.06) are numerical artefacts.
        # =========================================================
        raw_strehl = float(info.get("se_strehl", 0.0))
        current_strehl = float(np.clip(raw_strehl, 0.0, 1.0))

        # =========================================================
        # 6) Shaped reward
        # =========================================================
        action_norm = float(np.linalg.norm(rl_delta))  # ≤ sqrt(6)*DELTA_SCALE
        reward = _strehl_reward(current_strehl, action_norm)

        if self.current_step >= EPISODE_LEN:
            truncated = True

        # =========================================================
        # 7) Build info dict with corrected Strehl
        # =========================================================
        info = dict(info)
        info["se_strehl"]      = current_strehl   # overwrite with clamped value
        info["raw_se_strehl"]  = raw_strehl        # keep original for debugging
        info["mhe_increment"]  = mhe_increment
        info["rl_delta"]       = rl_delta
        info["gate_weight"]    = gate_weight
        info["final_action"]   = final_action
        info["x_hat"]          = x_hat

        return self._preprocess_obs(obs), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, num_episodes: int = 10, save_plot: bool = True):
    env = RLMHEEnv()

    scaled_rewards   = []
    raw_strehl_sums  = []
    final_strehls    = []
    all_episode_strehl = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)

        scaled_total = 0.0
        strehl_sum   = 0.0
        final_strehl = 0.0
        strehl_curve = []

        for _ in range(EPISODE_LEN):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            current_strehl = float(info.get("se_strehl", 0.0))  # already clamped

            scaled_total += reward
            strehl_sum   += current_strehl
            final_strehl  = current_strehl
            strehl_curve.append(current_strehl)

            if terminated or truncated:
                break

        scaled_rewards.append(scaled_total)
        raw_strehl_sums.append(strehl_sum)
        final_strehls.append(final_strehl)
        all_episode_strehl.append(strehl_curve)

        print(
            f"[RL+MHE] ep {ep + 1:2d} | "
            f"reward={scaled_total:7.2f} | "
            f"strehl_sum={strehl_sum:7.2f} | "
            f"final_strehl={final_strehl:.3f}"
        )

    print("\n=== RL+MHE FINAL SUMMARY ===")
    print(f"reward      : {np.mean(scaled_rewards):.2f} ± {np.std(scaled_rewards):.2f}")
    print(f"strehl_sum  : {np.mean(raw_strehl_sums):.2f} ± {np.std(raw_strehl_sums):.2f}")
    print(f"final_strehl: {np.mean(final_strehls):.3f} ± {np.std(final_strehls):.3f}")

    if save_plot:
        plt.figure(figsize=(10, 6))
        for i, curve in enumerate(all_episode_strehl):
            plt.plot(curve, label=f"Ep {i + 1}", alpha=0.7)
        plt.axvline(GATE_START, color="gray", linestyle="--", linewidth=1,
                    label=f"RL gate start (step {GATE_START})")
        plt.axvline(GATE_END,   color="black", linestyle="--", linewidth=1,
                    label=f"RL gate end (step {GATE_END})")
        plt.ylim(0.0, 1.05)
        plt.xlabel("Step")
        plt.ylabel("Strehl (clamped to [0, 1])")
        plt.title("RL+MHE Strehl Curves")
        plt.legend(fontsize=7, ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("rl_mhe_strehl_curves.png", dpi=200)
        print("\nSaved: rl_mhe_strehl_curves.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = RLMHEEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,      # small entropy bonus aids exploration during ramp
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
    )

    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    model.save("ppo_rl_mhe")

    evaluate_model(model, num_episodes=10, save_plot=True)