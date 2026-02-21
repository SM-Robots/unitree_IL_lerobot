"""Convert Matrix sim recordings to LeRobot v2 dataset format.

Our recorder (g1_control.recorder.DataRecorder) writes per-episode parquet files
with 29-DOF joint data plus separate hand arrays, and per-camera MP4 videos.
This script extracts the 14 upper-body arm joints + 14 hand joints = 28 DOF,
decodes video frames, and feeds everything through LeRobotDataset.create() /
add_frame() / save_episode().

Usage:
    python unitree_lerobot/utils/convert_matrix_to_lerobot.py \
        --recording-dir /path/to/recordings/episode_20260220_213422 \
        --repo-id local/sim_pick_cup \
        --fps 50
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from unitree_lerobot.utils.constants import ROBOT_CONFIGS

# ── FALCON 29-DOF joint ordering ──────────────────────────────────────────
#   0-5:   left leg
#   6-11:  right leg
#   12-14: waist
#   15-21: left arm  (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
#   22-28: right arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
UPPER_BODY_INDICES = list(range(15, 29))  # 14 arm DOFs


@dataclass
class ConvertArgs:
    recording_dir: str
    """Path to a Matrix recording directory (contains meta/, data/, videos/)."""
    repo_id: str = "local/sim_pick_cup"
    """LeRobot dataset repo-id (local/ prefix keeps it local)."""
    fps: int = 50
    """Recording FPS (must match the original recording rate)."""
    task: str = "Pick up the cup."
    """Task description string stored in every frame."""
    robot_type: str = "Unitree_G1_Dex3_Sim_Realsense"
    """Robot config key in constants.py."""
    cameras: list[str] = field(default_factory=lambda: ["realsense"])
    """Which cameras to include (must exist in videos/ dir)."""


def convert(args: ConvertArgs) -> None:
    rec_dir = Path(args.recording_dir)
    assert rec_dir.exists(), f"Recording dir not found: {rec_dir}"

    # ── Read episode list ─────────────────────────────────────────────────
    episodes_path = rec_dir / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    print(f"Found {len(episodes)} episodes in {rec_dir.name}")

    # ── Read info.json for video shapes ───────────────────────────────────
    info = json.loads((rec_dir / "meta" / "info.json").read_text())

    # ── Build features dict ───────────────────────────────────────────────
    motors = ROBOT_CONFIGS[args.robot_type].motors
    n_motors = len(motors)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (n_motors,),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (n_motors,),
            "names": [motors],
        },
    }

    for cam_name in args.cameras:
        cam_key = f"observation.images.{cam_name}"
        cam_info = info["features"].get(cam_key, {})
        shape = tuple(cam_info.get("shape", [360, 640, 3]))
        features[cam_key] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }

    # ── Create LeRobot dataset ────────────────────────────────────────────
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=4,
        image_writer_threads=4,
    )

    total_frames = 0

    for ep_meta in tqdm.tqdm(episodes, desc="Converting episodes"):
        ep_idx = ep_meta["episode_index"]

        # ── Read parquet ──────────────────────────────────────────────────
        pq_path = rec_dir / "data" / f"episode_{ep_idx:06d}.parquet"
        table = pq.read_table(pq_path)

        joint_pos = np.array(table.column("joint_pos").to_pylist(), dtype=np.float32)
        hand_left = np.array(table.column("hand_left").to_pylist(), dtype=np.float32)
        hand_right = np.array(table.column("hand_right").to_pylist(), dtype=np.float32)
        action_29 = np.array(table.column("action").to_pylist(), dtype=np.float32)

        n_steps = len(joint_pos)

        # ── Build 28-DOF vectors ──────────────────────────────────────────
        # State: measured arm positions + hand targets
        arm_state = joint_pos[:, UPPER_BODY_INDICES]  # (N, 14)
        state_28 = np.concatenate([arm_state, hand_left, hand_right], axis=1)  # (N, 28)

        # Action: commanded arm targets + hand targets
        arm_action = action_29[:, UPPER_BODY_INDICES]  # (N, 14)
        action_28 = np.concatenate([arm_action, hand_left, hand_right], axis=1)  # (N, 28)

        # ── Decode video frames ───────────────────────────────────────────
        cam_frames: dict[str, list[np.ndarray]] = {}
        for cam_name in args.cameras:
            video_path = (
                rec_dir / "videos" / f"observation.images.{cam_name}" / f"episode_{ep_idx:06d}.mp4"
            )
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if len(frames) != n_steps:
                print(
                    f"  Warning: ep {ep_idx} has {n_steps} parquet rows "
                    f"but {len(frames)} video frames ({cam_name})"
                )
            cam_frames[cam_name] = frames

        # Truncate to shortest length across all sources
        min_len = min(
            n_steps,
            *(len(cam_frames[c]) for c in args.cameras),
        )
        state_28 = state_28[:min_len]
        action_28 = action_28[:min_len]
        for c in args.cameras:
            cam_frames[c] = cam_frames[c][:min_len]

        # ── Add frames to dataset ─────────────────────────────────────────
        for i in range(min_len):
            frame = {
                "observation.state": state_28[i],
                "action": action_28[i],
                "task": args.task,
            }
            for cam_name in args.cameras:
                frame[f"observation.images.{cam_name}"] = cam_frames[cam_name][i]
            dataset.add_frame(frame)

        dataset.save_episode()
        total_frames += min_len

    print(f"\nConversion complete!")
    print(f"  Output:   {output_path}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Frames:   {total_frames}")
    print(f"  Motors:   {n_motors} ({', '.join(args.cameras)})")
    print(f"  FPS:      {args.fps}")


if __name__ == "__main__":
    convert(tyro.cli(ConvertArgs))
