"""Multi-agent MuJoCo on the Gymnasium/MuJoCo 3 stack, used only for the periodic evaluations during training.

Drop-in replacement for OMIGA's ``envs.ma_mujoco.multiagent_mujoco.mujoco_multi.MujocoMulti`` (gym 0.17.2 +
mujoco-py 2.0.2.8), reimplemented on gymnasium + mujoco 3.x. The observation semantics are replicated as OMIGA
produced them, because the released datasets were generated with that code:

  * every agent observes the FULL single-agent gym observation plus a one-hot agent id
    (OMIGA's obsk-based partial observation path is commented out in the original),
  * the resulting vector is normalized *within itself*: ``(x - x.mean()) / x.std()``,
  * ``state`` is identical to ``obs`` -- verified against the released hdf5.

Only the action partition comes from ``obsk.get_parts_and_edges``, which is pure numpy and is vendored unchanged.

Ant is the one scenario whose observation layout moved between MuJoCo versions: Ant-v2 included the (always-zero)
worldbody row of ``cfrc_ext``, giving 111 dims, while v5 dropped it, giving 105. ``_ANT_PAD`` reinserts that zero
block so the vector matches what the dataset was trained on. The contact forces themselves are not zero under
MuJoCo 3, so Ant returns from this environment are not comparable to the datasets; every return reported in the
paper comes from the MuJoCo 2.0 environment in mj200_env.py instead.
"""
import numpy as np
import gymnasium as gym

from .obsk import get_parts_and_edges

# Single-agent gym observation width that the released datasets assume (MuJoCo 2.00 "-v2").
_V2_OBS_DIM = {"Hopper-v2": 11, "HalfCheetah-v2": 17, "Ant-v2": 111}

# Modern gymnasium env id + kwargs reproducing the closest equivalent of each "-v2" env.
_MODERN_ENV = {
    "Hopper-v2": ("Hopper-v5", {}),
    "HalfCheetah-v2": ("HalfCheetah-v5", {}),
    "Ant-v2": ("Ant-v5", {"include_cfrc_ext_in_observation": True}),
}

# Ant-v2 laid out qpos[2:] (13) ++ qvel (14) ++ cfrc_ext[0:14] (84).  v5 emits
# cfrc_ext[1:14] only, so a 6-wide zero block is reinserted at index 27.
_ANT_PAD_AT = 27
_ANT_PAD_WIDTH = 6


class MujocoMulti:
    """Multi-agent wrapper matching OMIGA's MujocoMulti interface."""

    def __init__(self, batch_size=None, **kwargs):
        env_args = kwargs["env_args"]
        self.scenario = env_args["scenario"]          # e.g. "Hopper-v2"
        self.agent_conf = env_args["agent_conf"]      # e.g. "3x1"
        self.episode_limit = env_args.get("episode_limit", 1000)

        self.agent_partitions, self.mujoco_edges, self.mujoco_globals = get_parts_and_edges(
            self.scenario, self.agent_conf
        )
        self.n_agents = len(self.agent_partitions)
        self.n_actions = max(len(p) for p in self.agent_partitions)

        env_id, env_kwargs = _MODERN_ENV[self.scenario]
        self.env = gym.make(env_id, **env_kwargs).unwrapped
        self._np_random_seed = None
        self.steps = 0

        raw_dim = _V2_OBS_DIM[self.scenario]
        self.obs_size = raw_dim + self.n_agents
        self.share_obs_size = self.obs_size

        from gymnasium.spaces import Box

        self.observation_space = [
            Box(low=-10, high=10, shape=(self.obs_size,)) for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-10, high=10, shape=(self.share_obs_size,)) for _ in range(self.n_agents)
        ]
        acdims = [len(p) for p in self.agent_partitions]
        lo, hi = self.env.action_space.low, self.env.action_space.high
        self.action_space = tuple(
            Box(lo[sum(acdims[:a]):sum(acdims[: a + 1])],
                hi[sum(acdims[:a]):sum(acdims[: a + 1])])
            for a in range(self.n_agents)
        )
        self.n = self.n_agents

        self.env.reset(seed=self._np_random_seed)

    # ------------------------------------------------------------------ obs --
    def _raw_obs(self):
        """Single-agent observation in the -v2 layout the datasets were built from."""
        o = np.asarray(self.env._get_obs(), dtype=np.float64).ravel()
        if self.scenario == "Ant-v2":
            o = np.concatenate([o[:_ANT_PAD_AT], np.zeros(_ANT_PAD_WIDTH), o[_ANT_PAD_AT:]])
        assert o.size == _V2_OBS_DIM[self.scenario], (
            f"{self.scenario}: got {o.size} dims, dataset expects "
            f"{_V2_OBS_DIM[self.scenario]}"
        )
        return o

    def get_obs(self):
        state = self._raw_obs()
        obs_n = []
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            obs_i = np.concatenate([state, agent_id_feats])
            obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
            obs_n.append(obs_i)
        return obs_n

    # OMIGA's get_state() is identical to get_obs(); the released hdf5 has s == o.
    def get_state(self, team=None):
        return self.get_obs()

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.share_obs_size

    def get_avail_actions(self):
        return np.ones(shape=(self.n_agents, self.n_actions))

    def get_avail_agent_actions(self, agent_id):
        return np.ones(shape=(self.n_actions,))

    def get_total_actions(self):
        return self.n_actions

    # --------------------------------------------------------------- control --
    def _denormalize(self, action):
        """OMIGA wrapped the env in NormalizedActions: a in [-1,1] -> env range.

        For Hopper/HalfCheetah/Ant the env range is already [-1,1], so this is the
        identity; it is kept so other scenarios stay faithful.
        """
        lo, hi = self.env.action_space.low, self.env.action_space.high
        a = (np.asarray(action) + 1) / 2
        return a * (hi - lo) + lo

    def step(self, actions):
        # drop padding that arises when agents have unequal action widths
        flat = np.concatenate(
            [np.asarray(actions[i]).ravel()[: self.action_space[i].low.shape[0]]
             for i in range(self.n_agents)]
        )
        _, reward, terminated, truncated, info_n = self.env.step(self._denormalize(flat))
        self.steps += 1
        done = bool(terminated) or bool(truncated) or self.steps >= self.episode_limit

        info = dict(info_n)
        if done:
            # gym 0.17 folded the TimeLimit into `done`; keep OMIGA's masking flag
            info["bad_transition"] = self.steps >= self.episode_limit

        rewards = [[reward]] * self.n_agents
        dones = [done] * self.n_agents
        infos = [info for _ in range(self.n_agents)]
        return self.get_obs(), self.get_state(), rewards, dones, infos, self.get_avail_actions()

    def reset(self, **kwargs):
        self.steps = 0
        self.env.reset(seed=self._np_random_seed)
        self._np_random_seed = None  # seed only once, then continue the stream
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def seed(self, seed=None):
        self._np_random_seed = seed

    def close(self):
        self.env.close()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def get_stats(self):
        return {}

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "action_spaces": self.action_space,
            "actions_dtype": np.float32,
            "normalise_actions": False,
        }
