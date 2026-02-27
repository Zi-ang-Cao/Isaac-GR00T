#!/usr/bin/env python3
#!/usr/bin/env -S python -u
"""GR00T N1.5 inference server for SONIC (Unitree G1 latent actions).

This script loads a converted SONIC checkpoint using the public GR00T N1.5
codebase and exposes a ZMQ inference server with the same two-step API used
by the internal groot_dc_main policy server:

    set_observation(obs)   ->  stores + decodes observation
    get_action(time=None)  ->  runs inference on stored observation

Usage::

    # Convert the internal checkpoint first (one-time):
    python scripts/convert_sonic_checkpoint.py \\
        --input-dir /path/to/internal/checkpoint-20000 \\
        --output-dir /path/to/converted/checkpoint

    # Run the server:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_sonic_inference_server.py \\
        --model-path /path/to/converted/checkpoint \\
        --port 6666

Downstream clients connect with ZMQ and call::

    client.set_observation({
        "video.ego_view": [jpeg_bytes, ...],       # (T,) list of JPEG bytes
        "state.left_leg": np.ndarray,               # (T, D) or (B, T, D)
        "state.right_leg": np.ndarray,
        ...
        "annotation.human.task_description": "pick up the can",
    })
    action = client.get_action()
    # action = {"action.motion_token": np.ndarray, ...}
"""

import time as tm
from dataclasses import dataclass
from typing import Any, Dict, Literal

import cv2
import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.service import BaseInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


def _check_video_is_batched(video: np.ndarray) -> bool:
    if len(video) == 0:
        return False
    if video.dtype == np.object_:
        return video.ndim == 2
    return video.ndim == 5


def _decode_video(video_array: np.ndarray) -> np.ndarray:
    """Decode JPEG-encoded video bytes to numpy array."""
    if video_array.dtype in (
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32,
        np.float16, np.float32, np.float64,
    ):
        return video_array

    is_batched = _check_video_is_batched(video_array)
    if is_batched:
        return np.array([
            [cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR) for frame in batch]
            for batch in video_array
        ])
    else:
        return np.array([
            cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            for frame in video_array
        ])


class SonicInferenceServer(BaseInferenceServer):
    """ZMQ server with backward-compatible set_observation + get_action API."""

    def __init__(
        self,
        policy: Gr00tPolicy,
        host: str = "*",
        port: int = 6666,
    ):
        super().__init__(host=host, port=port)
        self.policy = policy
        self.observation = None
        self._time_start = tm.time()

        self.register_endpoint("set_observation", self.set_observation, requires_input=True)
        self.register_endpoint("get_action", self.get_action, requires_input=True)
        self.register_endpoint(
            "get_modality_config", self.get_modality_config, requires_input=False
        )

    def set_observation(self, observation: Dict[str, Any]):
        """Decode and store observation for the next get_action call."""
        self._time_start = tm.time()
        if observation is None:
            self.observation = None
            return

        for key, value in observation.items():
            if "video" in key:
                video_array = np.array(value) if not isinstance(value, np.ndarray) else value
                observation[key] = _decode_video(video_array)
            elif key.startswith("annotation.") or key.startswith("language."):
                observation[key] = value

        self.observation = observation

    def get_action(self, time: float | None = None) -> Dict[str, Any]:
        """Run inference on stored observation and return actions."""
        assert self.observation is not None, (
            "Observation not set -- call set_observation first"
        )

        action = self.policy.get_action(self.observation)
        elapsed = (tm.time() - self._time_start) * 1000
        print(f"Inference time: {elapsed:.1f} ms")

        return action

    def get_modality_config(self) -> Dict[str, Any]:
        return self.policy.get_modality_config()


@dataclass
class ServerConfig:
    model_path: str = ""
    """Path to the converted SONIC checkpoint directory."""

    port: int = 6666
    """Server port number."""

    host: str = "0.0.0.0"
    """Server host address."""

    data_config: str = "unitree_g1_sonic_latent"
    """Data config name (registered in data_config.py)."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = (
        "unitree_g1_whole_body_teleop_latent"
    )
    """Embodiment tag for the model."""

    denoising_steps: int = 4
    """Number of denoising steps."""

    device: str = "cuda"
    """Device to run inference on."""


def main(config: ServerConfig):
    if not config.model_path:
        raise ValueError("--model-path is required")

    print("=" * 70)
    print("  SONIC GR00T N1.5 Inference Server")
    print("=" * 70)
    print(f"  Model:      {config.model_path}")
    print(f"  Embodiment: {config.embodiment_tag}")
    print(f"  Data cfg:   {config.data_config}")
    print(f"  Port:       {config.port}")
    print(f"  Denoise:    {config.denoising_steps} steps")
    print("=" * 70)

    data_cfg = load_data_config(config.data_config)
    modality_config = data_cfg.modality_config()
    modality_transform = data_cfg.transform()

    policy = Gr00tPolicy(
        model_path=config.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=config.embodiment_tag,
        denoising_steps=config.denoising_steps,
        device=config.device,
    )

    server = SonicInferenceServer(policy, host=config.host, port=config.port)

    print(f"\nServer ready on tcp://{config.host}:{config.port}")
    print("Endpoints: set_observation, get_action, get_modality_config, ping, kill")
    print("Press Ctrl+C to stop.\n")

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
