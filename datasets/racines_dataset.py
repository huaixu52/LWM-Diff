"""
RACINES Dataset Loader for LWM-Diffusion
===================================
Cardiac ultrasound dataset with 5 standard views (A4C, SC, PL, PSAV, PSMV)
+ random Full scans. Each frame has image + 4x4 pose matrix + force/torque.

Data structure:
    /data/racines/
    ├── Folder 1/  (session 1)
    │   ├── A4C/
    │   │   ├── images/
    │   │   │   └── img*.png
    │   │   └── logs/
    │   │       └── PA_*.txt
    │   ├── SC/
    │   │   ├── images_1/, images_2/
    │   │   └── logs_1/, logs_2/
    │   └── ...
    ├── Folder 2/  (session 2)
    └── Folder 3/  (session 3)

Pose txt format:
    Time: <timestamp>
    Force and Torque: [[fx], [fy], [fz], [tx], [ty], [tz]]
    Pose Matrix (ROT):
    [[r11 r12 r13 x]
     [r21 r22 r23 y]
     [r31 r32 r33 z]
     [0   0   0   1]]
    Control Effort (CO): [vx vy vz wx wy wz]
"""

import os
import re
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from typing import List, Dict, Tuple, Optional


def parse_racines_pose_file(file_path: str) -> Dict:
    """
    Parse a RACINES pose txt file.

    Returns:
        dict with keys: 'timestamp', 'force_torque', 'pose_matrix', 'control_effort'
    """
    result = {
        "timestamp": None,
        "force_torque": None,
        "pose_matrix": None,
        "control_effort": None,
    }

    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Parse timestamp
        time_match = re.search(r"Time:\s*([\d.]+)", content)
        if time_match:
            result["timestamp"] = float(time_match.group(1))

        # Parse force and torque (6 values)
        ft_match = re.search(r"Force and Torque:\s*\[\[(.*?)\]\]", content, re.DOTALL)
        if ft_match:
            ft_str = (
                ft_match.group(1).replace("[", "").replace("]", "").replace("\n", " ")
            )
            result["force_torque"] = np.fromstring(ft_str, sep=" ", count=6)

        # Parse 4x4 pose matrix
        pose_match = re.search(
            r"Pose Matrix \(ROT\):\s*\[\[(.*?)\]\]", content, re.DOTALL
        )
        if pose_match:
            pose_str = (
                pose_match.group(1).replace("[", "").replace("]", "").replace("\n", " ")
            )
            result["pose_matrix"] = np.fromstring(pose_str, sep=" ", count=16).reshape(
                4, 4
            )

        # Parse control effort (6 values)
        co_match = re.search(r"Control Effort \(CO\):\s*\[(.*?)\]", content)
        if co_match:
            result["control_effort"] = np.fromstring(
                co_match.group(1), sep=" ", count=6
            )

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return result


def pose_matrix_to_6d(pose_matrix: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 pose matrix to 6D representation [x, y, z, euler_x, euler_y, euler_z].
    Euler angles in degrees.
    """
    try:
        # Validate shape
        if pose_matrix.ndim != 2 or pose_matrix.shape[0] < 4 or pose_matrix.shape[1] < 4:
            print(f"Warning: pose_matrix has unexpected shape {pose_matrix.shape}, returning zeros")
            return np.zeros(6)

        # Extract translation
        translation = pose_matrix[:3, 3].copy()

        # Extract rotation sub-matrix and validate
        rot_mat = pose_matrix[:3, :3].copy()
        if rot_mat.shape != (3, 3):
            print(f"Warning: rotation matrix has unexpected shape {rot_mat.shape}, returning zeros")
            return np.zeros(6)

        rotation = Rotation.from_matrix(rot_mat)
        euler = rotation.as_euler("xyz", degrees=True)

        return np.concatenate([translation, euler])
    except Exception as e:
        print(f"Warning: pose_matrix_to_6d failed ({e}), returning zeros")
        return np.zeros(6)


class RACINESDataset(Dataset):
    """
    RACINES Dataset for LWM-Diff disentanglement experiments.

    Args:
        root_dir: Path to RACINES root directory
        folders: List of folder names to include (e.g., ['Folder 1', 'Folder 2'])
        views: List of view names to include (e.g., ['A4C', 'SC', 'PL', 'PSAV', 'PSMV'])
                Use None for all views including 'Full'
        image_size: Target image size (default 256)
        return_pairs: If True, return ranked pairs (for original LWM-Diff training)
        normalize_pose: If True, normalize pose to [-1, 1] range
    """

    VIEW_NAMES = ["A4C", "SC", "PL", "PSAV", "PSMV", "Full"]
    VIEW_TO_IDX = {v: i for i, v in enumerate(VIEW_NAMES)}

    def __init__(
        self,
        root_dir: str,
        folders: List[str] = None,
        views: List[str] = None,
        image_size: int = 256,
        return_pairs: bool = False,
        normalize_pose: bool = True,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.folders = folders or ["Folder 1", "Folder 2", "Folder 3"]
        self.views = views or self.VIEW_NAMES
        self.image_size = image_size
        self.return_pairs = return_pairs
        self.normalize_pose = normalize_pose

        # Build index of all samples
        self.samples = []  # List of (image_path, pose_path, folder_idx, view_idx, traj_idx, frame_idx)
        self.trajectories = []  # List of trajectory info for splitting

        self._build_index()

        # Compute normalization stats if needed
        if self.normalize_pose:
            self._compute_pose_stats()

    def _build_index(self):
        """Scan dataset and build sample index."""
        traj_idx = 0

        for folder_idx, folder in enumerate(self.folders):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                print(f"Warning: {folder_path} not found, skipping")
                continue

            for view in self.views:
                view_path = os.path.join(folder_path, view)
                if not os.path.isdir(view_path):
                    continue

                view_idx = self.VIEW_TO_IDX.get(view, -1)

                # Find all image directories (images, images_1, images_2, or numbered subdirs for Full)
                image_dirs = []
                for d in os.listdir(view_path):
                    d_path = os.path.join(view_path, d)
                    if os.path.isdir(d_path):
                        if d.startswith("images"):
                            image_dirs.append((d_path, d.replace("images", "logs")))
                        elif d.isdigit():  # For Folder 3/Full/1, 2, 3
                            img_subdir = os.path.join(d_path, "images")
                            log_subdir = os.path.join(d_path, "logs")
                            if os.path.isdir(img_subdir):
                                image_dirs.append((img_subdir, log_subdir))

                if not image_dirs:
                    # Single images/ and logs/ directory
                    img_dir = os.path.join(view_path, "images")
                    log_dir = os.path.join(view_path, "logs")
                    if os.path.isdir(img_dir):
                        image_dirs.append((img_dir, log_dir))

                for img_dir, log_dir in image_dirs:
                    if isinstance(log_dir, str) and not os.path.isabs(log_dir):
                        log_dir = os.path.join(view_path, log_dir)

                    if not os.path.isdir(img_dir) or not os.path.isdir(log_dir):
                        continue

                    # Get all image files
                    img_files = glob.glob(os.path.join(img_dir, "*.png"))

                    traj_samples = []
                    for img_path in img_files:
                        # Extract frame number
                        basename = os.path.basename(img_path)
                        match = re.search(r"img(\d+)\.png", basename, re.I)
                        if not match:
                            continue
                        frame_num = int(match.group(1))

                        # Find corresponding pose file
                        pose_path = os.path.join(log_dir, f"PA_{frame_num}.txt")
                        if not os.path.exists(pose_path):
                            continue

                        traj_samples.append(
                            {
                                "image_path": img_path,
                                "pose_path": pose_path,
                                "folder_idx": folder_idx,
                                "view_idx": view_idx,
                                "traj_idx": traj_idx,
                                "frame_idx": frame_num,
                            }
                        )

                    if traj_samples:
                        # Sort by frame index
                        traj_samples.sort(key=lambda x: x["frame_idx"])
                        self.samples.extend(traj_samples)
                        self.trajectories.append(
                            {
                                "traj_idx": traj_idx,
                                "folder": folder,
                                "view": view,
                                "num_frames": len(traj_samples),
                            }
                        )
                        traj_idx += 1

        print(
            f"RACINES: Found {len(self.samples)} samples in {len(self.trajectories)} trajectories"
        )

    def _compute_pose_stats(self):
        """Compute mean and std for pose normalization."""
        poses = []
        for sample in self.samples[
            : min(1000, len(self.samples))
        ]:  # Sample subset for efficiency
            pose_data = parse_racines_pose_file(sample["pose_path"])
            if pose_data["pose_matrix"] is not None:
                pose_6d = pose_matrix_to_6d(pose_data["pose_matrix"])
                poses.append(pose_6d)

        if poses:
            poses = np.array(poses)
            self.pose_mean = poses.mean(axis=0)
            self.pose_std = poses.std(axis=0) + 1e-8
        else:
            self.pose_mean = np.zeros(6)
            self.pose_std = np.ones(6)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4
        )

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        # Load pose
        pose_data = parse_racines_pose_file(sample["pose_path"])

        if pose_data["pose_matrix"] is not None:
            pose_6d = pose_matrix_to_6d(pose_data["pose_matrix"])
        else:
            pose_6d = np.zeros(6)

        if self.normalize_pose:
            pose_6d = (pose_6d - self.pose_mean) / self.pose_std

        pose_6d = torch.from_numpy(pose_6d).float()

        # Get labels
        view_label = sample["view_idx"]
        folder_label = sample["folder_idx"]  # Domain label

        return {
            "image": image,
            "pose": pose_6d,
            "view_label": view_label,
            "domain_label": folder_label,
            "traj_idx": sample["traj_idx"],
            "frame_idx": sample["frame_idx"],
        }

    def get_domain_split(
        self, test_folders: List[str]
    ) -> Tuple["RACINESDataset", "RACINESDataset"]:
        """
        Split dataset by folder (domain) for cross-domain evaluation.

        Args:
            test_folders: Folders to use for testing (e.g., ['Folder 3'])

        Returns:
            train_dataset, test_dataset
        """
        train_folders = [f for f in self.folders if f not in test_folders]

        train_dataset = RACINESDataset(
            self.root_dir,
            folders=train_folders,
            views=self.views,
            image_size=self.image_size,
            return_pairs=self.return_pairs,
            normalize_pose=self.normalize_pose,
        )

        test_dataset = RACINESDataset(
            self.root_dir,
            folders=test_folders,
            views=self.views,
            image_size=self.image_size,
            return_pairs=False,
            normalize_pose=self.normalize_pose,
        )

        # Share normalization stats
        test_dataset.pose_mean = train_dataset.pose_mean
        test_dataset.pose_std = train_dataset.pose_std

        return train_dataset, test_dataset


if __name__ == "__main__":
    # Test the dataset
    dataset = RACINESDataset(
        root_dir="/data/racines/",
        folders=["Folder 1", "Folder 2", "Folder 3"],
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Trajectories: {len(dataset.trajectories)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Pose shape: {sample['pose'].shape}")
    print(f"View label: {sample['view_label']}")
    print(f"Domain label: {sample['domain_label']}")
