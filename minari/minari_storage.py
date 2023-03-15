import os
from typing import Any, Callable, Iterable, List, Optional, Union
import gymnasium as gym
import h5py
from gymnasium.envs.registration import EnvSpec


_PathLike = Union[str, bytes, os.PathLike]


class MinariStorage:
    def __init__(self, data_path: _PathLike):
        """Initialize properties of the Minari storage.

        Args:
            data_path (str): full path to the `main_data.hdf5` file of the dataset.
        """
        self._data_path = data_path
        self._extra_data_id = 0
        with h5py.File(self._data_path, "r") as f:
            self._flatten_observations = f.attrs["flatten_observation"]
            self._flatten_actions = f.attrs["flatten_action"]
            self._env_spec = EnvSpec.from_json(f.attrs["env_spec"])

            _total_episodes = f.attrs["total_episodes"]
            assert isinstance(_total_episodes, int)
            self._total_episodes: int = _total_episodes

            _total_steps = f.attrs["total_steps"]
            assert isinstance(_total_steps, int)
            self._total_steps: int = _total_steps

            self._dataset_name = f.attrs["dataset_name"]
            self._combined_datasets = f.attrs.get("combined_datasets")

            env = gym.make(self._env_spec)

            self._observation_space = env.observation_space
            self._action_space = env.action_space

            env.close()

    def apply(
        self,
        function: Callable[[h5py.Group], Any],
        episode_indices: Optional[Iterable] = None
    ):
        if episode_indices is None:
            episode_indices = range(self.total_episodes)
        out = []
        with h5py.File(self._data_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)
                out.append(function(ep_group))

        return out
    
    def get_episodes(self, episode_indices: Iterable) -> List[dict]:
        out = []
        with h5py.File(self._data_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                out.append(dict(ep_group.items()))

        return out

    @property
    def flatten_observations(self) -> bool:
        """If the observations have been flatten when creating the dataset."""
        return self._flatten_observations

    @property
    def flatten_actions(self) -> bool:
        """If the actions have been flatten when creating the dataset."""
        return self._flatten_actions

    @property
    def observation_space(self):
        """Original observation space of the environment before flatteining (if this is the case)."""
        return self._observation_space

    @property
    def action_space(self):
        """Original action space of the environment before flatteining (if this is the case)."""
        return self._action_space

    @property
    def data_path(self):
        """Full path to the `main_data.hdf5` file of the dataset."""
        return self._data_path

    @property
    def total_steps(self):
        """Total steps recorded in the Minari dataset along all episodes."""
        return self._total_steps

    @property
    def total_episodes(self):
        """Total episodes recorded in the Minari dataset."""
        return self._total_episodes

    @property
    def env_spec(self):
        """EnvSpec of the environment that has generated the dataset."""
        return self._env_spec

    @property
    def combined_datasets(self):
        """If this Minari dataset is a combination of other subdatasets, return a list with the subdataset names."""
        if self._combined_datasets is None:
            return []
        else:
            return self._combined_datasets

    @property
    def name(self):
        """Name of the Minari dataset."""
        return self._dataset_name
