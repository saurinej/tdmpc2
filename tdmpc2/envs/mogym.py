from typing import Any, Dict, Tuple, SupportsFloat, Union

import gymnasium
import mo_gymnasium as mo_gym

from envs.wrappers.time_limit import TimeLimit


TASKS = {
    "lunar_lander": {
        "id": "mo-lunar-lander-continuous-v2",
        "reward_config": {
            "crashed_or_landed": {
                "index": 0
            },
            "shaping":{
                "index": 1,
                "scale_if_done": 0.0
            },
            "main_engine_fuel": {
                "index": 2,
                "scale": 0.3, # from env step method
                "scale_if_done": 0.0
            },
            "side_engine_fuel": {
                "index": 3,
                "scale": 0.03, # from env step method
                "scale_if_done": 0.0
            }
        }
    },
    "four_room": {
        "id": "four-room-v0",
        "reward_config": {
            "blue_square": {
                "index": 0
            },
            "green_triangle": {
                "index": 1
            },
            "red_circle": {
                "index": 2
            }
        }
    },
    "half_cheetah": {
        "id": "mo-halfcheetah-v4",
        "reward_config": {
            "run_forward": {
                "index": 0
            },
            "control_cost": {
                "index": 1
            }
        }
    }
}


class ScalarMOGymWrapper(gymnasium.Wrapper):
    """Wrapper for a mo-gymansium environment. Instead of return reward as a numpy array, we
    return the scalar reward and add the component rewards to the info dict individually.

    The names for the componenet rewards are specified in the component_reward_config key in
    the environment configuration.

    The component reward config should include the name of the reward as the dict key. In the
    dict, there should be an 'index' key to specify which index in the mo gym array the reward
    name corresponds to. Optionally, there is a 'scale' key to multiply the component reward by.
    This is used in a case like lunar lander where the step function adds component rewards that
    are not scaled the same individually as in the scalar reward.

    See sheeprl.configs.env.mo_lundar_lander.yaml as an example.
    """
    def __init__(self, id: str, reward_config: Dict[str, Any], seed: Union[int, None] = None, **kwargs) -> None:

        env = mo_gym.make(env_name=id, **kwargs)

        super().__init__(env)

        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        self.reward_config = reward_config

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        # call step on mo gym env
        obs, reward, terminated, truncated, info = self.env.step(action)

        # scale component rewards based on config and add to info
        # some envs scale some rewards to zero if the episode is done (like lunar lander)
        # this is on the person running the env to figure out and add the necessary scale
        # and scale_if_done keys
        scale_key = 'scale_if_done' if terminated else 'scale'
        for k, v in self.reward_config.items():
            info[k] = reward[v['index']] * v.get(scale_key, 1.0)

        # set reward to actual reward
        reward = info.get('original_reward', sum(reward))

        return obs, reward, terminated or truncated, info
    
    def reset(self):
        return self.env.reset()[0]

def make_env(cfg):
    """
    Make MO-Gymnasium environment.
    """
    assert cfg.task in TASKS
    task = TASKS[cfg.task]
    env = ScalarMOGymWrapper(task["id"], task["reward_config"], cfg.seed)
    env.max_episode_steps = 1000
    return env
