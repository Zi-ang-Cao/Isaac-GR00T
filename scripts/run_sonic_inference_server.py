#!/usr/bin/env python3
"""GR00T N1.5 inference server for SONIC (Unitree G1 latent actions).

Wire-compatible with the internal groot_dc_main GrootN1ClientPolicy client:
uses the same TorchSerializer (torch.save/load over ZMQ) and the same
two-step API (set_observation -> get_action).

Usage::

    # Prepare the checkpoint (one-time, non-destructive):
    python scripts/convert_sonic_checkpoint.py \\
        --input-dir /path/to/internal/checkpoint-20000 --prepare

    # Run the server:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_sonic_inference_server.py \\
        --model-path /path/to/internal/checkpoint-20000 \\
        --port 6666
"""

import time as tm
import traceback
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Dict, Literal

import cv2
import numpy as np
import torch
import tyro
import zmq

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


# ---------------------------------------------------------------------------
# TorchSerializer -- identical to groot_dc_main/groot/control/utils/service.py
# This is what the internal GrootN1ClientPolicy speaks.
# ---------------------------------------------------------------------------

class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buf = BytesIO()
        torch.save(data, buf)
        return buf.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return torch.load(BytesIO(data), weights_only=False)


# ---------------------------------------------------------------------------
# Minimal ZMQ server using TorchSerializer (matches the internal protocol)
# ---------------------------------------------------------------------------

@dataclass
class _EndpointHandler:
    handler: Callable
    requires_input: bool = True


class TorchZmqServer:
    """ZMQ REP server using TorchSerializer, matching the internal protocol."""

    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, _EndpointHandler] = {}
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        self.running = False

    def _handle_ping(self) -> dict:
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        self._endpoints[name] = _EndpointHandler(handler, requires_input)

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                traceback.print_exc()
                self.socket.send(b"ERROR")


# ---------------------------------------------------------------------------
# Video decoding helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SONIC inference server
# ---------------------------------------------------------------------------

class SonicInferenceServer(TorchZmqServer):
    """ZMQ server with backward-compatible set_observation + get_action API.

    Uses TorchSerializer so the internal GrootN1ClientPolicy can connect
    without any client-side changes.
    """

    def __init__(
        self,
        policy: Gr00tPolicy,
        host: str = "*",
        port: int = 6666,
        video_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        language_keys: list[str] | None = None,
    ):
        super().__init__(host=host, port=port)
        self.policy = policy
        self.observation = None
        self._time_start = tm.time()

        self._video_keys = set(video_keys or [])
        self._state_keys = set(state_keys or [])
        self._language_keys = set(language_keys or [])
        self._accepted_keys = self._video_keys | self._state_keys | self._language_keys

        self.register_endpoint("set_observation", self.set_observation, requires_input=True)
        self.register_endpoint("get_action", self.get_action, requires_input=True)
        self.register_endpoint(
            "get_modality_config", self.get_modality_config, requires_input=False
        )

    def set_observation(self, observation: Dict[str, Any]):
        """Decode, filter, and store observation for the next get_action call.

        The internal client sends extra keys (e.g. "q") and annotation as a
        bare string.  We filter to only the keys the data config expects and
        wrap language values in a list so the transform pipeline can batch them.
        """
        self._time_start = tm.time()
        if observation is None:
            self.observation = None
            return

        clean_obs: Dict[str, Any] = {}
        for key, value in observation.items():
            if self._accepted_keys and key not in self._accepted_keys:
                continue

            if key in self._video_keys:
                video_array = np.array(value) if not isinstance(value, np.ndarray) else value
                clean_obs[key] = _decode_video(video_array)
            elif key in self._language_keys:
                if isinstance(value, str):
                    clean_obs[key] = [value]
                else:
                    clean_obs[key] = value
            else:
                clean_obs[key] = value

        self.observation = clean_obs

    def get_action(self, data: dict | None = None) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    model_path: str = ""
    """Path to the SONIC checkpoint directory (run --prepare first)."""

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
    print(f"  Serializer: TorchSerializer (internal-client compatible)")
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

    server = SonicInferenceServer(
        policy,
        host=config.host,
        port=config.port,
        video_keys=data_cfg.video_keys,
        state_keys=data_cfg.state_keys,
        language_keys=data_cfg.language_keys,
    )

    print(f"\nEndpoints: set_observation, get_action, get_modality_config, ping, kill")
    print("Press Ctrl+C to stop.\n")

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
