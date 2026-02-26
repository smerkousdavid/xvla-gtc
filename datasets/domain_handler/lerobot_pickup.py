from __future__ import annotations

import random

import numpy as np
import torch
from mmengine import fileio
from PIL import Image
from scipy.interpolate import interp1d

from ..utils import read_parquet, read_video_to_frames
from .base import DomainHandler


class LeRobotPickupHandler(DomainHandler):
    """
    Handler for LeRobot v2.1-style single-arm datasets with columns:
      - actions: [T, 7]
      - state: [T, 7] (optional; actions are used as fallback)
    and camera videos stored as:
      videos/chunk-XXX/{image,wrist_image}/episode_XXXXXX.mp4
    """

    CAMERA_VIEW = ["image", "wrist_image"]
    ACTION_KEY = "actions"
    STATE_KEY = "state"
    idx_for_delta = [0, 1, 2, 3, 4, 5]
    idx_for_mask_proprio = [6]
    # Hardcoded per-dataset, per-channel scales for raw action channels.
    # Verified with pyarrow scan:
    # - insert-blender ch06 max=240
    # - insert-mujoco ch06 max=240
    # - insert_centrifuge_5430-blender ch06 max=240
    # Keep xarm-lab-data untouched (handled in LeRobotXArmLabHandler).
    ACTION_CHANNEL_SCALES_BY_DATASET = {
        "insert-blender": {6: 240.0},
        "insert-mujoco": {6: 240.0},
        "insert_centrifuge_5430-blender": {6: 240.0},
    }

    def _scale_raw_action_channels(
        self, actions: np.ndarray, states: np.ndarray, dataset_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        channel_scales = self.ACTION_CHANNEL_SCALES_BY_DATASET.get(dataset_name)
        if not channel_scales:
            return actions, states

        dim = int(actions.shape[1])
        for gi, scale in channel_scales.items():
            gi = int(gi)
            scale = float(scale)
            if gi < 0 or gi >= dim or scale <= 0:
                continue
            actions[:, gi] = actions[:, gi] / scale
            states[:, gi] = states[:, gi] / scale
        return actions, states

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

        data = read_parquet(data_path)
        if self.ACTION_KEY not in data:
            raise KeyError(f"Missing '{self.ACTION_KEY}' in {data_path}")

        all_action = np.asarray(data[self.ACTION_KEY], dtype=np.float32)
        if all_action.ndim != 2:
            raise ValueError(f"Expected action shape [T, D], got {all_action.shape} in {data_path}")

        # Proprio from state if available, otherwise from actions.
        if self.STATE_KEY in data:
            all_state = np.asarray(data[self.STATE_KEY], dtype=np.float32)
            if all_state.shape[0] != all_action.shape[0]:
                all_state = all_action.copy()
        else:
            all_state = all_action.copy()

        dataset_name = str(self.meta.get("dataset_name", ""))
        all_action, all_state = self._scale_raw_action_channels(all_action, all_state, dataset_name)

        # Match training convention used in existing handlers: shift by one.
        all_action = np.concatenate([all_action[:1], all_action[:-1]], axis=0)

        fps = float(self.meta.get("fps", 30.0))
        qdur = 1.0
        t = np.arange(all_action.shape[0], dtype=np.float64) / max(fps, 1e-8)

        if all_action.shape[0] < 4:
            return

        idxs = list(range(1, all_action.shape[0] - 1))
        if training:
            random.shuffle(idxs)

        action_interp = interp1d(
            t,
            all_action,
            axis=0,
            bounds_error=False,
            fill_value=(all_action[0], all_action[-1]),
        )
        state_interp = interp1d(
            t,
            all_state,
            axis=0,
            bounds_error=False,
            fill_value=(all_state[0], all_state[-1]),
        )

        instruction = item["tasks"][0] if item.get("tasks") else "Perform the task."

        for idx in idxs:
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

            # Build abs trajectory [H+1, 20] with first row as proprio/state.
            abs_traj = np.zeros((num_actions + 1, 20), dtype=np.float32)
            d = min(seq_action.shape[-1], 20)
            abs_traj[0, :d] = seq_state0[:d]
            abs_traj[1:, :d] = seq_action[1:, :d]

            cur_action = torch.from_numpy(abs_traj)
            if (cur_action[1] - cur_action[0]).abs().max() < 1e-5:
                continue

            if lang_aug_map is not None and instruction in lang_aug_map:
                instruction = random.choice(lang_aug_map[instruction])

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
