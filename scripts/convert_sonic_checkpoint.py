#!/usr/bin/env python3
"""Convert an internal GrootN1d5 checkpoint for the public GR00T_N1_5 repo.

Two modes of operation:

  --prepare   (recommended, non-destructive)
      Writes config_public.json and experiment_cfg/metadata.json *alongside*
      the original config.json.  The checkpoint stays loadable by both repos.

  --output-dir <path>
      Copies weights to a new directory with a rewritten config.json.

Usage:

    # Non-destructive prepare (no weight copy, keeps config.json intact):
    python scripts/convert_sonic_checkpoint.py \\
        --input-dir /path/to/internal/checkpoint-20000 --prepare

    # Full copy to a separate directory:
    python scripts/convert_sonic_checkpoint.py \\
        --input-dir /path/to/internal/checkpoint-20000 \\
        --output-dir /path/to/converted/checkpoint
"""

import argparse
import json
import shutil
from pathlib import Path


def convert_config(input_config: dict) -> dict:
    """Restructure flat GrootN1d5Config into nested GR00T_N1_5_Config."""

    backbone_embedding_dim = input_config.get("backbone_embedding_dim", 2048)

    # The internal model uses nn.Identity() when project_to_dim == 2048,
    # while the public code creates nn.Linear when project_to_dim is not None.
    # Set to None to get Identity behavior and avoid missing-key errors.
    project_to_dim = None if backbone_embedding_dim == 2048 else backbone_embedding_dim

    backbone_cfg = {
        "tune_llm": input_config.get("tune_llm", False),
        "tune_visual": input_config.get("tune_visual", False),
        "select_layer": input_config.get("select_layer", 12),
        "reproject_vision": input_config.get("reproject_vision", False),
        "use_flash_attention": input_config.get("use_flash_attention", True),
        "load_bf16": input_config.get("load_bf16", False),
        "eagle_path": None,
        "project_to_dim": project_to_dim,
    }

    action_dim = input_config.get("max_action_dim", 128)
    action_horizon = input_config.get("action_horizon", 50)

    action_head_cfg = {
        "add_pos_embed": input_config.get("add_pos_embed", True),
        "model_dtype": input_config.get("model_dtype", "bfloat16"),
        "diffusion_model_cfg": input_config.get("diffusion_model_cfg", {}),
        "input_embedding_dim": input_config.get("input_embedding_dim", 1536),
        "backbone_embedding_dim": backbone_embedding_dim,
        "hidden_size": input_config.get("hidden_size", 1024),
        "max_seq_len": input_config.get("max_seq_len", 1024),
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "noise_beta_alpha": input_config.get("noise_beta_alpha", 1.5),
        "noise_beta_beta": input_config.get("noise_beta_beta", 1.0),
        "noise_s": input_config.get("noise_s", 0.999),
        "num_timestep_buckets": input_config.get("num_timestep_buckets", 1000),
        "num_inference_timesteps": input_config.get("num_inference_timesteps", 4),
        "max_num_embodiments": input_config.get("max_num_embodiments", 32),
        "tune_projector": input_config.get("tune_projector", True),
        "tune_diffusion_model": input_config.get("tune_diffusion_model", True),
        "use_vlln": input_config.get("use_vlln", True),
        "use_future_tokens": input_config.get("use_future_tokens", False),
        "vl_self_attention_cfg": input_config.get("vl_self_attention_cfg", {}),
        "max_state_dim": input_config.get("max_state_dim", 64),
    }

    return {
        "model_type": "gr00t_n1_5",
        "architectures": ["GR00T_N1_5"],
        "backbone_cfg": backbone_cfg,
        "action_head_cfg": action_head_cfg,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "compute_dtype": input_config.get("model_dtype", "bfloat16"),
        "torch_dtype": input_config.get("torch_dtype", "bfloat16"),
        "transformers_version": input_config.get("transformers_version", "4.51.3"),
    }


def convert_statistics_to_metadata(
    stats_path: Path,
    embodiment_tag: str = "unitree_g1_whole_body_teleop_latent",
    video_resolution: tuple[int, int] = (640, 480),
) -> dict:
    """Convert internal dataset_statistics.json to public metadata.json format."""
    with open(stats_path) as f:
        all_stats = json.load(f)

    emb_stats = all_stats.get(embodiment_tag, {})
    state_stats = emb_stats.get("state", {})
    action_stats = emb_stats.get("action", {})

    state_modalities = {}
    for key, vals in state_stats.items():
        dim = len(vals["mean"])
        state_modalities[key] = {
            "absolute": True,
            "rotation_type": None,
            "shape": (dim,),
            "continuous": True,
        }

    action_modalities = {}
    for key, vals in action_stats.items():
        dim = len(vals["mean"])
        action_modalities[key] = {
            "absolute": True,
            "rotation_type": None,
            "shape": (dim,),
            "continuous": True,
        }

    video_modalities = {
        "ego_view": {
            "resolution": list(video_resolution),
            "channels": 3,
            "fps": 50.0,
        }
    }

    return {
        embodiment_tag: {
            "statistics": {"state": state_stats, "action": action_stats},
            "modalities": {
                "video": video_modalities,
                "state": state_modalities,
                "action": action_modalities,
            },
            "embodiment_tag": embodiment_tag,
        }
    }


def _create_metadata(
    target_dir: Path,
    stats_path: Path,
    embodiment_tag: str,
    video_resolution: tuple[int, int] = (640, 480),
):
    """Write experiment_cfg/metadata.json from dataset_statistics.json."""
    exp_cfg_dir = target_dir / "experiment_cfg"
    exp_cfg_dir.mkdir(exist_ok=True)
    if stats_path.exists():
        metadata = convert_statistics_to_metadata(stats_path, embodiment_tag, video_resolution)
        with open(exp_cfg_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Created experiment_cfg/metadata.json for '{embodiment_tag}'")
        print(f"    video resolution: {video_resolution[0]}x{video_resolution[1]}")
    else:
        print(f"  WARNING: {stats_path} not found, skipping metadata generation")


def prepare_in_place(input_dir: Path, embodiment_tag: str, video_resolution: tuple[int, int]):
    """Non-destructive: write config_public.json + metadata.json, never touch config.json."""
    print(f"Preparing checkpoint (non-destructive): {input_dir}")

    with open(input_dir / "config.json") as f:
        internal_config = json.load(f)

    converted = convert_config(internal_config)
    public_cfg_path = input_dir / "config_public.json"
    with open(public_cfg_path, "w") as f:
        json.dump(converted, f, indent=2)
    print(f"  Wrote {public_cfg_path.name}")
    print(f"    model_type: {internal_config.get('model_type')} -> {converted['model_type']}")
    print(f"    action_dim={converted['action_dim']}  action_horizon={converted['action_horizon']}")

    stats_path = input_dir / "experiment_cfg" / "dataset_statistics.json"
    _create_metadata(input_dir, stats_path, embodiment_tag, video_resolution)

    print(f"\nDone. config.json is untouched -- checkpoint is loadable by both repos.")


def copy_to_output(input_dir: Path, output_dir: Path, embodiment_tag: str, video_resolution: tuple[int, int]):
    """Full copy: write converted config.json + weights to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Converting config.json ...")
    with open(input_dir / "config.json") as f:
        internal_config = json.load(f)
    converted = convert_config(internal_config)
    with open(output_dir / "config.json", "w") as f:
        json.dump(converted, f, indent=2)
    print(f"  model_type: {internal_config.get('model_type')} -> {converted['model_type']}")
    print(f"  action_dim={converted['action_dim']}  action_horizon={converted['action_horizon']}")

    print("Copying model weights ...")
    for sf_file in input_dir.glob("model*.safetensors"):
        shutil.copy2(sf_file, output_dir / sf_file.name)
        print(f"  {sf_file.name}")
    index_file = input_dir / "model.safetensors.index.json"
    if index_file.exists():
        shutil.copy2(index_file, output_dir / index_file.name)
        print(f"  {index_file.name}")

    stats_path = input_dir / "experiment_cfg" / "dataset_statistics.json"
    _create_metadata(output_dir, stats_path, embodiment_tag, video_resolution)

    for extra in ["experiment_cfg/conf.yaml", "experiment_cfg/dataset_statistics.json"]:
        src = input_dir / extra
        if src.exists():
            dst = output_dir / extra
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print(f"\nConversion complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert internal GrootN1d5 checkpoint for the public GR00T_N1_5 repo"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Path to internal checkpoint")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Non-destructive: write config_public.json + metadata.json alongside config.json",
    )
    mode.add_argument(
        "--output-dir",
        type=str,
        help="Full copy: write converted checkpoint to this directory",
    )

    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="unitree_g1_whole_body_teleop_latent",
    )
    parser.add_argument(
        "--video-resolution",
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=("W", "H"),
        help="Native camera resolution (width height). Default: 640 480",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    video_res = tuple(args.video_resolution)

    if args.prepare:
        prepare_in_place(input_dir, args.embodiment_tag, video_res)
    else:
        copy_to_output(input_dir, Path(args.output_dir), args.embodiment_tag, video_res)


if __name__ == "__main__":
    main()
