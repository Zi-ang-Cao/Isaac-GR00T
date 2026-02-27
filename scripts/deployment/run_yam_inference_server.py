"""Lightweight GR00T inference server for the XDOF (YAM) robot.

This script loads a local checkpoint, creates a Gr00tPolicy with optional
torch.compile and custom denoising steps, wraps it for real-robot observation
formats (serialized JPEG video over ZMQ), and starts the server.

When --use-torch-compile is enabled the script always runs a warmup inference
before accepting connections. TorchInductor cache lives under
TORCHINDUCTOR_CACHE_DIR (default: /tmp/torchinductor_$USER/) and is reused
across processes as long as model/code/environment fingerprints match.

- First launch (or after cache/key changes): compiles and writes cache.
- Later launches: should hit FX/AOT caches, but first request can still take
  noticeable time for one-time graph/module loading and runtime initialization.

The "Server ready" message only appears once warmup completes, so the server
is guaranteed to respond at full speed from the very first request.

Usage::

    cd Isaac-GR00T
    source .venv/bin/activate

    python scripts/deployment/run_yam_inference_server.py \
        --embodiment-tag XDOF \
        --use-torch-compile \
        --num-inference-timesteps 16 \
        /path/to/local/checkpoint/

The robot control loop connects separately::

    uv run python -m yam_control.launch --mode server-eval \
        --gr00t-host <this-machine-ip> --gr00t-port 5555
"""

import logging
import os
import socket
import threading
import time
from typing import Annotated, Literal

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tRealRobotPolicyWrapper
from gr00t.policy.server_client import PolicyServer

DEFAULT_PORT = 5555


def _warmup_policy(policy: Gr00tPolicy, embodiment_tag: EmbodimentTag) -> None:
    """Run dummy inference to trigger torch.compile JIT compilation.

    Times each phase of the inference pipeline individually so we can identify
    the bottleneck (preprocessing, backbone, torch.compiled action head, etc).
    """
    import torch
    from torch._dynamo.utils import counters as dynamo_counters
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    from gr00t.data.types import MessageType, VLAStepData

    resolved_cache_dir = cache_dir()
    print(f"[Warmup] Inductor cache dir: {resolved_cache_dir}")
    print(
        "[Warmup] Running warmup inference with per-phase timing and "
        "torch.compile cache diagnostics..."
    )

    modality_configs = policy.processor.get_modality_configs()[embodiment_tag.value]
    video_keys = modality_configs["video"].modality_keys
    state_keys = modality_configs["state"].modality_keys
    language_keys = modality_configs["language"].modality_keys
    norm_params = policy.processor.state_action_processor.norm_params[embodiment_tag.value]

    dummy_video = {
        key: np.zeros((1, 1, 240, 320, 3), dtype=np.uint8) for key in video_keys
    }
    dummy_state = {}
    for key in state_keys:
        dim = len(norm_params["state"][key]["mean"])
        dummy_state[key] = np.zeros((1, 1, dim), dtype=np.float32)
    dummy_language = {key: [["warmup"]] for key in language_keys}
    dummy_obs = {"video": dummy_video, "state": dummy_state, "language": dummy_language}

    t_total = time.time()

    # Phase 1: Preprocessing (unbatch + VLA processor + collate)
    t0 = time.time()
    unbatched = policy._unbatch_observation(dummy_obs)
    obs = unbatched[0]
    vla_step = VLAStepData(
        images=obs["video"],
        states=obs["state"],
        actions={},
        text=obs["language"][policy.language_key][0],
        embodiment=embodiment_tag,
    )
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
    processed = policy.processor(messages)
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    collated = policy.collate_fn([processed])
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    print(f"[Warmup]   Preprocessing:  {time.time() - t0:.1f}s", flush=True)

    # Phase 2: Full model forward (backbone + action head)
    print(
        "[Warmup]   Model forward #1: running... (torch.compile may compile or load cache)",
        flush=True,
    )

    # Snapshot counters before first compiled invocation.
    inductor_before = dict(dynamo_counters.get("inductor", {}))
    aot_before = dict(dynamo_counters.get("aot_autograd", {}))

    stop_ticker = threading.Event()
    t1 = time.time()

    def _progress_ticker():
        while not stop_ticker.wait(15):
            elapsed = time.time() - t1
            print(f"[Warmup]   Model forward #1: ... {elapsed:.0f}s elapsed", flush=True)

    ticker = threading.Thread(target=_progress_ticker, daemon=True)
    ticker.start()

    with torch.inference_mode():
        model_pred = policy.model.get_action(**collated)
    stop_ticker.set()
    ticker.join()

    model_time = time.time() - t1
    print(f"[Warmup]   Model forward #1: {model_time:.1f}s", flush=True)

    # Snapshot counters after first compiled invocation.
    inductor_after = dict(dynamo_counters.get("inductor", {}))
    aot_after = dict(dynamo_counters.get("aot_autograd", {}))

    def _delta(after: dict[str, int], before: dict[str, int], key: str) -> int:
        return int(after.get(key, 0)) - int(before.get(key, 0))

    fx_hit = _delta(inductor_after, inductor_before, "fxgraph_cache_hit")
    fx_miss = _delta(inductor_after, inductor_before, "fxgraph_cache_miss")
    fx_bypass = _delta(inductor_after, inductor_before, "fxgraph_cache_bypass")
    aot_hit = _delta(aot_after, aot_before, "autograd_cache_hit")
    aot_miss = _delta(aot_after, aot_before, "autograd_cache_miss")
    aot_bypass = _delta(aot_after, aot_before, "autograd_cache_bypass")

    print(
        "[Warmup]   Cache counters: "
        f"fx(hit={fx_hit}, miss={fx_miss}, bypass={fx_bypass}), "
        f"aot(hit={aot_hit}, miss={aot_miss}, bypass={aot_bypass})",
        flush=True,
    )

    # Phase 2b: immediate second call to show post-warm steady-state latency.
    t1b = time.time()
    with torch.inference_mode():
        _ = policy.model.get_action(**collated)
    model_time_second = time.time() - t1b
    print(f"[Warmup]   Model forward #2: {model_time_second:.1f}s", flush=True)

    # Phase 3: Action decoding
    t2 = time.time()
    normalized_action = model_pred["action_pred"].float()
    batched_states = {}
    for k in modality_configs["state"].modality_keys:
        batched_states[k] = obs["state"][k][np.newaxis]
    policy.processor.decode_action(
        normalized_action.cpu().numpy(), embodiment_tag, batched_states
    )
    print(f"[Warmup]   Action decode:  {time.time() - t2:.1f}s", flush=True)

    total = time.time() - t_total
    print(f"[Warmup] Total: {total:.1f}s", flush=True)

    # Report cache status from counters instead of using a wall-clock heuristic.
    if (fx_hit > 0 or aot_hit > 0) and (
        fx_miss == 0 and aot_miss == 0 and fx_bypass == 0 and aot_bypass == 0
    ):
        print("[Warmup] << CACHE HIT -- loaded FX/AOT artifacts from disk")
    elif fx_hit > 0 or aot_hit > 0:
        print("[Warmup] << PARTIAL CACHE HIT -- some graphs reused, some rebuilt/bypassed")
    elif fx_miss > 0 or aot_miss > 0:
        print("[Warmup] << CACHE MISS -- compiled graphs and wrote cache entries")
    elif fx_bypass > 0 or aot_bypass > 0:
        print("[Warmup] << CACHE BYPASS -- graph(s) were not cacheable")
    else:
        print("[Warmup] << CACHE STATUS UNKNOWN -- no explicit hit/miss counters reported")

    if model_time_second < max(1.0, model_time * 0.2):
        print(
            "[Warmup] Note: A slow first call with fast second call usually indicates "
            "one-time loading/initialization, not full recompilation."
        )


def main(
    model_path: Annotated[str, tyro.conf.Positional],
    embodiment_tag: EmbodimentTag = EmbodimentTag.XDOF,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    device: str = "cuda",
    use_torch_compile: bool = False,
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
    num_inference_timesteps: int | None = None,
):
    """Start a GR00T inference server from a local checkpoint."""
    os.environ["HF_HUB_OFFLINE"] = "1"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    hostname = socket.gethostname()
    print("\n" + "=" * 70)
    print("  GR00T Inference Server (open-source)")
    print("=" * 70)
    print(f"  Host:       {hostname}")
    print(f"  Port:       {port}")
    print(f"  Model:      {model_path}")
    print(f"  Embodiment: {embodiment_tag.value}")
    print(f"  Compile:    {use_torch_compile} (mode={torch_compile_mode})")
    print(f"  Timesteps:  {num_inference_timesteps or 'model default'}")
    print()
    print("  Connect from yam-control:")
    print(f"    uv run python -m yam_control.launch --mode server-eval \\")
    print(f"        --gr00t-host {hostname} --gr00t-port {port}")
    print("=" * 70 + "\n")

    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=model_path,
        device=device,
        num_inference_timesteps=num_inference_timesteps,
        use_torch_compile=use_torch_compile,
        torch_compile_mode=torch_compile_mode,
    )

    if use_torch_compile:
        _warmup_policy(policy, embodiment_tag)

    policy = Gr00tRealRobotPolicyWrapper(policy)

    server = PolicyServer(policy=policy, host=host, port=port)

    try:
        print(f"\nServer ready and listening on {host}:{port}. Press Ctrl+C to stop.")
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    tyro.cli(main)
