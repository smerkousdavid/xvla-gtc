#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from datasets.utils import read_parquet, read_video_to_frames
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor


def parse_args():
    ap = argparse.ArgumentParser("Chunked restart open-loop eval")
    ap.add_argument("--model", type=str, required=True, help="Checkpoint dir or HF repo")
    ap.add_argument("--metas_path", type=str, required=True, help="Path to one generated meta json")
    ap.add_argument("--output_dir", type=str, default="open_loop_eval_episode", help="Output dir")
    ap.add_argument("--num_episodes", type=int, default=8, help="Randomly sampled episode count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk_steps", type=int, default=16, help="Predict K steps per restart")
    ap.add_argument("--num_rollouts", type=int, default=3, help="Rollouts sampled per chunk")
    ap.add_argument("--steps", type=int, default=10, help="Denoising steps for generate_actions")
    ap.add_argument("--plot_dims", type=int, default=8)
    ap.add_argument("--action_mode", type=str, default="auto")
    ap.add_argument("--real_action_dim", type=int, default=14)
    ap.add_argument("--max_action_dim", type=int, default=20)
    ap.add_argument(
        "--delta_indices",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated indices interpreted as relative-to-start deltas",
    )
    return ap.parse_args()


def choose_plot_dims(valid_d: int, max_plot_dims: int) -> list[int]:
    preferred = [0, 1, 2, 3, 4, 5]
    if valid_d > 6:
        preferred.append(6)
    if valid_d > 13:
        preferred.append(13)
    for i in range(valid_d):
        if i not in preferred:
            preferred.append(i)
    return preferred[: min(max_plot_dims, valid_d)]


def masked_mse_mae_np(pred: np.ndarray, target: np.ndarray, valid_dim: int) -> tuple[float, float]:
    d = min(pred.shape[-1], target.shape[-1], int(valid_dim))
    if d <= 0:
        return 0.0, 0.0
    md = min(pred.shape[-2], target.shape[-2])
    diff = pred[..., :md, :d] - target[..., :md, :d]
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    return mse, mae


def parse_indices_csv(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def to_absolute_from_start(rel_seq: np.ndarray, start_state: np.ndarray, delta_idx: list[int]) -> np.ndarray:
    # rel_seq: [T, D], start_state: [D]
    abs_seq = rel_seq.copy()
    d = abs_seq.shape[-1]
    for i in delta_idx:
        if 0 <= i < d:
            abs_seq[:, i] = abs_seq[:, i] + start_state[i]
    return abs_seq


def build_eval_image_aug():
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
    )


def resolve_instruction(task_map: dict, row_task_idx: int | None, fallback: str) -> str:
    if row_task_idx is None:
        return fallback
    if row_task_idx in task_map:
        return str(task_map[row_task_idx])
    if str(row_task_idx) in task_map:
        return str(task_map[str(row_task_idx)])
    return fallback


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.metas_path).open("r") as f:
        meta = json.load(f)
    datalist = list(meta.get("datalist", []))
    if not datalist:
        raise RuntimeError(f"No episodes in meta: {args.metas_path}")

    sampled = rng.sample(datalist, k=min(args.num_episodes, len(datalist)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XVLA.from_pretrained(
        args.model,
        action_mode=args.action_mode,
        real_action_dim=args.real_action_dim,
        max_action_dim=args.max_action_dim,
    ).to(device)
    processor = XVLAProcessor.from_pretrained(args.model)
    model.eval()
    image_aug = build_eval_image_aug()

    task_map = meta.get("task_map", {}) or {}
    delta_idx = parse_indices_csv(args.delta_indices)
    all_episode_rows = []

    for ep_item in tqdm(sampled, desc="Episodes"):
        ep_idx = int(ep_item["episode_index"])
        chunk_size = int(meta.get("chunks_size", 1000))
        ep_chunk = ep_idx // chunk_size
        ep_name = f"episode_{ep_idx:06d}"
        ep_out = out_dir / f"{ep_name}"
        ep_out.mkdir(parents=True, exist_ok=True)

        data_path = (meta.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        video_path = (meta.get("video_path") or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
        root_path = str(meta.get("root_path", ""))

        pq_path = (
            Path(root_path)
            / data_path.format(episode_chunk=ep_chunk, episode_index=ep_idx, episode_chunk_index=ep_chunk)
        )
        data = read_parquet(str(pq_path))

        # Copy zed_gripper video for visual verification.
        try:
            zed_video = (
                Path(root_path)
                / video_path.format(
                    episode_chunk=ep_chunk,
                    episode_index=ep_idx,
                    video_key="observation.images.zed_gripper",
                )
            )
            if zed_video.exists():
                shutil.copy2(zed_video, ep_out / zed_video.name)
        except Exception:
            pass

        # Load only the two expected camera views for xarm.
        cam_keys = ["observation.images.zed_gripper", "observation.images.zed_high_left"]
        frames = []
        for ck in cam_keys:
            vp = (
                Path(root_path)
                / video_path.format(
                    episode_chunk=ep_chunk,
                    episode_index=ep_idx,
                    video_key=ck,
                )
            )
            frames.append(read_video_to_frames(str(vp)))

        actions = np.asarray(data["action"], dtype=np.float32)
        states = np.asarray(data["observation.state"], dtype=np.float32)
        T = min(len(actions), len(states), len(frames[0]), len(frames[1]))
        if T <= 1:
            continue

        valid_dim = min(actions.shape[-1], args.max_action_dim)
        fallback_instruction = ep_item["tasks"][0] if ep_item.get("tasks") else "Perform the task."

        episode_chunk_rows = []
        # For plotting: one subplot per dim; GT black; each rollout colored by rollout id.
        plot_dims = choose_plot_dims(valid_dim, args.plot_dims)
        fig, axes = plt.subplots(len(plot_dims), 1, figsize=(12, 2.2 * len(plot_dims)), sharex=True)
        if len(plot_dims) == 1:
            axes = [axes]

        color_cycle = ["tab:red", "tab:blue", "tab:green"]
        for cstep in tqdm(range(0, T, args.chunk_steps), desc=f"Episode {ep_idx}", leave=False):
            end = min(cstep + args.chunk_steps, T)
            L = end - cstep
            if L <= 0:
                continue

            # Build per-chunk single observation (camera + proprio) then replicate for rollout batch.
            imgs = [image_aug(Image.fromarray(fr[cstep])) for fr in frames]
            image_input_1 = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)  # [V,C,H,W]
            image_mask_1 = torch.tensor([True, True], dtype=torch.bool, device=device)

            proprio = np.zeros((args.max_action_dim,), dtype=np.float32)
            d0 = min(states.shape[-1], args.max_action_dim)
            proprio[:d0] = states[cstep, :d0]
            # match train behavior for 7D single-arm: mask gripper in proprio
            if d0 > 6:
                proprio[6] = 0.0
            proprio_1 = torch.from_numpy(proprio).to(device=device)

            row_task_idx = None
            if "task_index" in data:
                try:
                    row_task_idx = int(data["task_index"][cstep])
                except Exception:
                    row_task_idx = None
            instruction = resolve_instruction(task_map, row_task_idx, fallback_instruction)

            lang = processor.encode_language([instruction] * args.num_rollouts)
            input_ids = lang["input_ids"].to(device=device)

            # replicate observation inputs for N rollout samples
            image_input = image_input_1.unsqueeze(0).repeat(args.num_rollouts, 1, 1, 1, 1)
            image_mask = image_mask_1.unsqueeze(0).repeat(args.num_rollouts, 1)
            proprio_b = proprio_1.unsqueeze(0).repeat(args.num_rollouts, 1)
            domain_val = int(data["domain_id"][cstep]) if "domain_id" in data else 0
            domain_id = torch.full((args.num_rollouts,), domain_val, dtype=torch.long, device=device)

            with torch.no_grad():
                pred = model.generate_actions(
                    input_ids=input_ids,
                    image_input=image_input,
                    image_mask=image_mask,
                    domain_id=domain_id,
                    proprio=proprio_b,
                    steps=args.steps,
                ).detach().float().cpu().numpy()  # [N, H, D]

            gt_abs = actions[cstep:end, :valid_dim]  # [L, D]
            start_abs = states[cstep, :valid_dim]
            # Clip predictions to available GT window length.
            pred_rel = pred[:, :L, :valid_dim]
            pred_abs = np.stack(
                [to_absolute_from_start(pred_rel[rid], start_abs, delta_idx) for rid in range(args.num_rollouts)],
                axis=0,
            )

            rollout_mses = []
            rollout_maes = []
            for rid in range(args.num_rollouts):
                mse_r, mae_r = masked_mse_mae_np(pred_abs[rid], gt_abs, valid_dim)
                rollout_mses.append(mse_r)
                rollout_maes.append(mae_r)

            chunk_row = {
                "episode_index": ep_idx,
                "chunk_start": int(cstep),
                "chunk_end": int(end),
                "chunk_len": int(L),
                "rollout_mse": rollout_mses,
                "rollout_mae": rollout_maes,
                "chunk_mse_avg": float(np.mean(rollout_mses)),
                "chunk_mae_avg": float(np.mean(rollout_maes)),
            }
            episode_chunk_rows.append(chunk_row)

            # Plot chunk trajectories in stitched timeline.
            x = np.arange(cstep, end, dtype=np.int32)
            for ax_i, dim_i in enumerate(plot_dims):
                # GT
                axes[ax_i].plot(x, gt_abs[:, dim_i], color="black", linewidth=1.6, alpha=0.8)
                # Marker where GT state is fed in (chunk boundary).
                axes[ax_i].scatter(
                    [cstep],
                    [states[cstep, dim_i] if dim_i < states.shape[1] else 0.0],
                    marker="o",
                    s=28,
                    facecolors="none",
                    edgecolors="black",
                )
                # Rollouts
                for rid in range(args.num_rollouts):
                    md = min(pred_abs[rid, :, dim_i].shape[0], x.shape[0])
                    axes[ax_i].plot(
                        x[:md],
                        pred_abs[rid, :md, dim_i],
                        color=color_cycle[rid % len(color_cycle)],
                        linewidth=1.1,
                        alpha=0.9,
                    )

        if not episode_chunk_rows:
            plt.close(fig)
            continue

        ep_mse = float(np.mean([r["chunk_mse_avg"] for r in episode_chunk_rows]))
        ep_mae = float(np.mean([r["chunk_mae_avg"] for r in episode_chunk_rows]))
        ep_row = {
            "episode_index": ep_idx,
            "traj_length": int(T),
            "num_chunks": int(len(episode_chunk_rows)),
            "chunk_steps": int(args.chunk_steps),
            "num_rollouts": int(args.num_rollouts),
            "episode_mse": ep_mse,
            "episode_mae": ep_mae,
        }
        all_episode_rows.append(ep_row)

        with (ep_out / "chunk_metrics.jsonl").open("w") as f:
            for r in episode_chunk_rows:
                f.write(json.dumps(r) + "\n")
        with (ep_out / "episode_metrics.json").open("w") as f:
            json.dump(ep_row, f, indent=2)

        for ax_i, dim_i in enumerate(plot_dims):
            axes[ax_i].set_ylabel(f"dim {dim_i}")
            if ax_i == 0:
                axes[ax_i].plot([], [], color="black", label="GT")
                for rid in range(args.num_rollouts):
                    axes[ax_i].plot([], [], color=color_cycle[rid % len(color_cycle)], label=f"pred_{rid}")
                axes[ax_i].legend(loc="upper right")
        axes[-1].set_xlabel("timestep")
        fig.suptitle(
            f"episode={ep_idx} chunk={args.chunk_steps} rollouts={args.num_rollouts} "
            f"MSE={ep_mse:.6f} MAE={ep_mae:.6f}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(ep_out / "trajectory_chunks.png", dpi=150)
        plt.close(fig)

    if not all_episode_rows:
        raise RuntimeError("No episodes evaluated. Check metas/model compatibility.")

    mse_arr = np.array([r["episode_mse"] for r in all_episode_rows], dtype=np.float64)
    mae_arr = np.array([r["episode_mae"] for r in all_episode_rows], dtype=np.float64)
    summary = {
        "num_episodes": int(len(all_episode_rows)),
        "chunk_steps": int(args.chunk_steps),
        "num_rollouts": int(args.num_rollouts),
        "episode_mse_mean": float(mse_arr.mean()),
        "episode_mse_median": float(np.median(mse_arr)),
        "episode_mse_p90": float(np.percentile(mse_arr, 90)),
        "episode_mae_mean": float(mae_arr.mean()),
        "episode_mae_median": float(np.median(mae_arr)),
        "episode_mae_p90": float(np.percentile(mae_arr, 90)),
    }
    with (out_dir / "metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "metrics_per_episode.jsonl").open("w") as f:
        for r in all_episode_rows:
            f.write(json.dumps(r) + "\n")

    plt.figure(figsize=(8, 5))
    plt.hist(mse_arr, bins=20)
    plt.title("Per-episode MSE distribution")
    plt.xlabel("Episode MSE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "mse_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(mae_arr, bins=20)
    plt.title("Per-episode MAE distribution")
    plt.xlabel("Episode MAE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "mae_hist.png", dpi=150)
    plt.close()

    print("Done.")
    print(f"- Output dir: {out_dir}")
    print(f"- Episodes evaluated: {len(all_episode_rows)}")
    print(
        f"- Episode MSE mean/median/p90: "
        f"{summary['episode_mse_mean']:.6f} / {summary['episode_mse_median']:.6f} / {summary['episode_mse_p90']:.6f}"
    )
    print(
        f"- Episode MAE mean/median/p90: "
        f"{summary['episode_mae_mean']:.6f} / {summary['episode_mae_median']:.6f} / {summary['episode_mae_p90']:.6f}"
    )


if __name__ == "__main__":
    main()
