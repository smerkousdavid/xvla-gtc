from __future__ import annotations

import random

import numpy as np
import torch
from mmengine import fileio
from PIL import Image
from scipy.interpolate import interp1d

from ..utils import read_parquet, read_video_to_frames
from .base import DomainHandler


class LeRobotXArmLabHandler(DomainHandler):
    """
    Handler for xarm-lab-data LeRobot dataset.

    Special requirements:
    - Use only two camera views:
        - observation.images.zed_gripper
        - observation.images.zed_high_left
    - Resolve language from parquet task_index + meta task_map when available.
    """

    CAMERA_VIEW = [
        "observation.images.zed_gripper",
        "observation.images.zed_high_left",
    ]
    ACTION_KEY = "action"
    STATE_KEY = "observation.state"
    TASK_INDEX_KEY = "task_index"
    TASK_DESC_KEY = "annotation.human.action.task_description"
    idx_for_delta = [0, 1, 2, 3, 4, 5]
    idx_for_mask_proprio = [6]

    def _resolve_instruction(self, data: dict, row_idx: int, fallback: str) -> str:
        task_map = self.meta.get("task_map", {}) or {}

        task_idx = None
        if self.TASK_INDEX_KEY in data:
            try:
                task_idx = int(data[self.TASK_INDEX_KEY][row_idx])
            except Exception:
                task_idx = None
        if task_idx is None and self.TASK_DESC_KEY in data:
            # In this dataset this field is int-like and mirrors task_index.
            try:
                task_idx = int(data[self.TASK_DESC_KEY][row_idx])
            except Exception:
                task_idx = None

        if task_idx is not None:
            if task_idx in task_map:
                return str(task_map[task_idx])
            if str(task_idx) in task_map:
                return str(task_map[str(task_idx)])

        return fallback

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        **kwargs,
    ):
        item = self.meta["datalist"][traj_idx]
        episode_index = int(item["episode_index"])
        chunk_size = int(self.meta.get("chunks_size", 1000))
        episode_chunk = episode_index // chunk_size

        data_path = fileio.join_path(self.meta["root_path"], self.meta["data_path"]).format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )
        data = read_parquet(data_path)
        if self.ACTION_KEY not in data:
            raise KeyError(f"Missing '{self.ACTION_KEY}' in {data_path}")
        if self.STATE_KEY not in data:
            raise KeyError(f"Missing '{self.STATE_KEY}' in {data_path}")

        actions = np.asarray(data[self.ACTION_KEY], dtype=np.float32)
        states = np.asarray(data[self.STATE_KEY], dtype=np.float32)
        if actions.ndim != 2 or states.ndim != 2:
            raise ValueError(f"Unexpected action/state dims in {data_path}: {actions.shape}, {states.shape}")

        images = [
            read_video_to_frames(
                fileio.join_path(self.meta["root_path"], self.meta["video_path"]).format(
                    episode_chunk=episode_chunk, episode_index=episode_index, video_key=vkey
                )
            )
            for vkey in self.CAMERA_VIEW
        ]
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[: min(self.num_views, len(images))] = True

        # Keep behavior consistent with other handlers: action shifted by one.
        actions = np.concatenate([actions[:1], actions[:-1]], axis=0)

        fps = float(self.meta.get("fps", 20.0))
        qdur = 1.0
        t = np.arange(actions.shape[0], dtype=np.float64) / max(fps, 1e-8)
        if actions.shape[0] < 4:
            return

        idxs = list(range(1, actions.shape[0] - 1))
        if training:
            random.shuffle(idxs)

        action_interp = interp1d(
            t, actions, axis=0, bounds_error=False, fill_value=(actions[0], actions[-1])
        )
        state_interp = interp1d(
            t, states, axis=0, bounds_error=False, fill_value=(states[0], states[-1])
        )

        fallback_instruction = item["tasks"][0] if item.get("tasks") else "Perform the task."

        for idx in idxs:
            instruction = self._resolve_instruction(data, idx, fallback_instruction)
            if lang_aug_map is not None and instruction in lang_aug_map:
                instruction = random.choice(lang_aug_map[instruction])

            imgs = []
            for v in range(min(self.num_views, len(images))):
                imgs.append(image_aug(Image.fromarray(images[v][idx])))
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, 0)

            cur = t[idx]
            q = np.linspace(cur, min(cur + qdur, float(t.max())), num_actions + 1, dtype=np.float32)
            seq_action = np.asarray(action_interp(q), dtype=np.float32)
            seq_state0 = np.asarray(state_interp(q[0]), dtype=np.float32)

            abs_traj = np.zeros((num_actions + 1, 20), dtype=np.float32)
            d = min(seq_action.shape[-1], 20)
            abs_traj[0, :d] = seq_state0[:d]
            abs_traj[1:, :d] = seq_action[1:, :d]

            cur_action = torch.from_numpy(abs_traj)
            if (cur_action[1] - cur_action[0]).abs().max() < 1e-5:
                continue

            yield {
                "language_instruction": instruction,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": cur_action.float(),
                "valid_action_dim": int(d),
                "episode_index": int(episode_index),
                "anchor_index": int(idx),
                "idx_for_delta": self.idx_for_delta,
                "idx_for_mask_proprio": self.idx_for_mask_proprio,
            }
