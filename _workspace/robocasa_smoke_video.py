from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio


def _patch_robosuite_seed() -> None:
    import inspect

    import robosuite
    from robosuite.environments.base import REGISTERED_ENVS

    original_make = robosuite.make

    def patched_make(*args, **kwargs):
        env_name = kwargs.get("env_name")
        if env_name in REGISTERED_ENVS:
            env_cls = REGISTERED_ENVS[env_name]
            sig = inspect.signature(env_cls.__init__)
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key == "env_name" or key in sig.parameters
            }
        return original_make(*args, **kwargs)

    robosuite.make = patched_make


def _pick_image(obs: dict) -> str:
    for key, value in obs.items():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) == 3 and shape[-1] in (1, 3, 4):
            return key
    for key in obs:
        if key.endswith("_image"):
            return key
    raise KeyError("No *_image observation key found")


def _patch_camera_config() -> None:
    from robocasa.utils.gym_utils import gymnasium_basic

    original_make_key_converter = gymnasium_basic.make_key_converter

    def patched_make_key_converter(robots_name):
        converter = original_make_key_converter(robots_name)
        original_get_camera_config = converter.get_camera_config

        def patched_get_camera_config():
            robot_name, camera_names, camera_widths, camera_heights = original_get_camera_config()
            mapped = ["agentview" if name == "egoview" else name for name in camera_names]
            return robot_name, mapped, camera_widths, camera_heights

        converter.get_camera_config = patched_get_camera_config
        return converter

    gymnasium_basic.make_key_converter = patched_make_key_converter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    _patch_robosuite_seed()

    import gymnasium as gym
    import robocasa  # noqa: F401
    import robosuite  # noqa: F401
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

    _patch_camera_config()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, enable_render=True)
    writer = imageio.get_writer(out_path, fps=10)
    try:
        obs, info = env.reset()
        image_key = _pick_image(obs)
        writer.append_data(obs[image_key])
        for step_idx in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            writer.append_data(obs[image_key])
            if terminated or truncated:
                break
    finally:
        writer.close()
        env.close()

    print(f"saved_video={out_path}")


if __name__ == "__main__":
    main()
