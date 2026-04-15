"""
Unified Dataset Wrapper and Utilities for LMW-Diffusion
==================================================
Provides a unified interface for RACINES and UltraBones100k datasets,
with support for BC (Behavior Cloning) training with 6D pose as action labels.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from typing import List, Dict, Tuple, Optional, Union


class UnifiedUSDataset(Dataset):
    """
    Unified wrapper that provides consistent interface across different datasets.

    Maps dataset-specific fields to unified names:
        - 'image': ultrasound image (1, H, W)
        - 'pose': 6D pose [x, y, z, euler_x, euler_y, euler_z]
        - 'content_label': semantic label (view for RACINES, anatomy for UltraBones)
        - 'domain_label': domain/patient label (folder for RACINES, specimen for UltraBones)
        - 'dataset_id': which dataset this sample comes from
    """

    def __init__(
        self,
        dataset,
        dataset_name: str,
        content_key: str = None,
        domain_key: str = None,
    ):
        """
        Args:
            dataset: The underlying dataset (RACINESDataset or UltraBonesDataset)
            dataset_name: Name identifier ('racines' or 'ultrabones')
            content_key: Key for content label (auto-detected if None)
            domain_key: Key for domain label (auto-detected if None)
        """
        self.dataset = dataset
        self.dataset_name = dataset_name

        # Auto-detect keys based on dataset type
        if content_key is None:
            self.content_key = (
                "view_label" if "racines" in dataset_name.lower() else "anatomy_label"
            )
        else:
            self.content_key = content_key

        if domain_key is None:
            self.domain_key = "domain_label"
        else:
            self.domain_key = domain_key

        # Dataset ID for multi-dataset training
        self.dataset_id = 0 if "racines" in dataset_name.lower() else 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {
            "image": sample["image"],
            "pose": sample["pose"],
            "content_label": sample.get(self.content_key, -1),
            "domain_label": sample.get(self.domain_key, -1),
            "dataset_id": self.dataset_id,
            "traj_idx": sample.get("traj_idx", -1),
            "frame_idx": sample.get("frame_idx", sample.get("timestamp", -1)),
        }


class SequenceBCDataset(Dataset):
    """
    Sequence-aware BC dataset for POMDP ultrasound navigation.

    Unlike BCDataset which feeds a single frame, this returns a sliding window
    of *obs_horizon* past observations and an action chunk of *action_horizon*
    future delta poses.  The observation history lets a temporal encoder
    disambiguate partial observations; the action chunk enables smoother and
    more consistent motion primitives.

    Returns:
        images : (obs_horizon, C, H, W) — observation window
        poses  : (obs_horizon, 6)        — corresponding poses (for probes)
        actions: (action_horizon, 6)      — future action chunk
        mask   : (obs_horizon,)           — 1 where real, 0 where padded
        + content_label, domain_label, traj_idx scalars
    """

    def __init__(
        self,
        dataset: Dataset,
        obs_horizon: int = 4,
        action_horizon: int = 4,
        use_delta_pose: bool = True,
        ood_corruption_type: Optional[str] = None,
        ood_corruption_severity: int = 3,
    ):
        self.dataset = dataset
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.use_delta_pose = use_delta_pose
        self.ood_corruption_type = ood_corruption_type
        self.ood_corruption_severity = ood_corruption_severity

        self._build_valid_indices()

    # ------------------------------------------------------------------
    def _build_valid_indices(self):
        """Group samples by trajectory and create valid (window, chunk) tuples."""
        traj_samples: Dict[int, List[int]] = {}
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            t_idx = sample.get("traj_idx", 0)
            frame_idx = sample.get("frame_idx", sample.get("timestamp", idx))
            if t_idx not in traj_samples:
                traj_samples[t_idx] = []
            traj_samples[t_idx].append((frame_idx, idx))

        self.valid_indices: List[Tuple[List[int], List[int], int]] = []
        for t_idx, samples in traj_samples.items():
            samples.sort(key=lambda x: x[0])
            indices = [i for _, i in samples]
            n = len(indices)
            # current position *t* can be 0 .. n - action_horizon - 1
            for t in range(n - self.action_horizon):
                # observation window: indices[t - obs_horizon + 1 .. t] (left-pad if needed)
                obs_start = max(0, t - self.obs_horizon + 1)
                obs_ids = indices[obs_start : t + 1]
                # action chunk: indices[t+1 .. t+action_horizon]
                act_ids = indices[t + 1 : t + 1 + self.action_horizon]
                self.valid_indices.append((obs_ids, act_ids, t_idx))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        obs_ids, act_ids, traj_idx = self.valid_indices[idx]

        # --- Observation window (left-pad with first frame if window < obs_horizon) ---
        obs_samples = [self.dataset[i] for i in obs_ids]
        pad_len = self.obs_horizon - len(obs_samples)

        images_list = [s["image"] for s in obs_samples]
        poses_list = [s["pose"] for s in obs_samples]

        if pad_len > 0:
            images_list = [images_list[0]] * pad_len + images_list
            poses_list = [poses_list[0]] * pad_len + poses_list

        images = torch.stack(images_list, dim=0)       # (obs_horizon, C, H, W)
        if self.ood_corruption_type is not None:
            from datasets.ood_corruptions import apply_ood_corruption
            images = torch.stack(
                [apply_ood_corruption(images[t], self.ood_corruption_type, self.ood_corruption_severity) for t in range(images.size(0))],
                dim=0,
            )
        poses = torch.stack(poses_list, dim=0)          # (obs_horizon, 6)
        mask = torch.cat([
            torch.zeros(pad_len),
            torch.ones(len(obs_samples)),
        ])                                              # (obs_horizon,)

        # --- Action chunk ---
        act_samples = [self.dataset[i] for i in act_ids]
        curr_pose = obs_samples[-1]["pose"]  # pose at time t

        if self.use_delta_pose:
            actions = torch.stack(
                [s["pose"] - curr_pose for s in act_samples], dim=0
            )  # (action_horizon, 6)
        else:
            actions = torch.stack(
                [s["pose"] for s in act_samples], dim=0
            )

        return {
            "images": images,            # (T_obs, C, H, W)
            "poses": poses,              # (T_obs, 6)
            "actions": actions,           # (T_act, 6)
            "mask": mask,                 # (T_obs,)
            "content_label": obs_samples[-1].get(
                "content_label", obs_samples[-1].get("view_label", -1)
            ),
            "domain_label": obs_samples[-1].get("domain_label", -1),
            "traj_idx": traj_idx,
            "frame_idx": obs_samples[-1].get(
                "frame_idx", obs_samples[-1].get("timestamp", -1)
            ),
            # keep single-frame aliases for backward compat with viz code
            "image": images[-1],
            "pose": curr_pose,
        }


class LWMSequenceDataset(SequenceBCDataset):
    """Sequence dataset that additionally returns future frame images.

    Extends :class:`SequenceBCDataset` with a ``future_images`` field
    containing the raw (un-augmented) images at future timesteps.  This
    is used by the Latent World Model (LWM) policy to compute
    stop-gradient target features for self-supervised latent prediction.
    """

    def __getitem__(self, idx):
        # Reuse parent logic for obs window, actions, metadata
        result = super().__getitem__(idx)

        # Retrieve future-frame images (no augmentation for target consistency)
        _, act_ids, _ = self.valid_indices[idx]
        future_images = torch.stack(
            [self.dataset[i]["image"] for i in act_ids], dim=0
        )  # (action_horizon, C, H, W)

        result["future_images"] = future_images
        return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = True,
    **kwargs,
) -> DataLoader:
    """Create DataLoader with default settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs,
    )


def create_cross_domain_splits(
    dataset_class, root_dir: str, test_domains: List, **dataset_kwargs
) -> Tuple[Dataset, Dataset]:
    """
    Create train/test splits based on domain (folder/specimen).

    Args:
        dataset_class: RACINESDataset or UltraBonesDataset
        root_dir: Path to dataset root
        test_domains: Domains to use for testing
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        train_dataset, test_dataset
    """
    full_dataset = dataset_class(root_dir=root_dir, **dataset_kwargs)

    if hasattr(full_dataset, "get_domain_split"):
        return full_dataset.get_domain_split(test_domains)
    elif hasattr(full_dataset, "get_specimen_split"):
        return full_dataset.get_specimen_split(test_domains)
    else:
        raise ValueError(f"Dataset {dataset_class} does not support domain splitting")


def _interleaved_val_test_split(
    dataset, val_ratio: float = 0.2, stride: int = 5, seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into val and test by interleaved sampling within each
    trajectory.  Every *stride*-th frame (determined per-trajectory) goes to
    val; the rest goes to test.

    This ensures:
      1. Val and test cover the same trajectory distribution (all views).
      2. Adjacent frames never both end up in val, avoiding near-duplicate leakage.
      3. The temporal spread within each trajectory is preserved in both splits.

    Args:
        dataset: base dataset (must expose traj_idx and frame_idx per sample)
        val_ratio: approximate fraction of samples for validation
        stride: take 1-in-stride frames for val (actual ratio ≈ 1/stride)
        seed: random seed for deterministic offset

    Returns:
        val_subset, test_subset  (torch.utils.data.Subset instances)
    """
    from torch.utils.data import Subset

    # Group indices by trajectory
    traj_to_indices: Dict[int, List[Tuple[int, int]]] = {}  # traj_idx -> [(frame_idx, dataset_idx)]
    for idx in range(len(dataset)):
        sample = dataset[idx]
        t_idx = sample.get("traj_idx", 0)
        f_idx = sample.get("frame_idx", sample.get("timestamp", idx))
        if t_idx not in traj_to_indices:
            traj_to_indices[t_idx] = []
        traj_to_indices[t_idx].append((f_idx, idx))

    rng = np.random.default_rng(seed)
    val_indices = []
    test_indices = []

    for t_idx in sorted(traj_to_indices.keys()):
        items = sorted(traj_to_indices[t_idx], key=lambda x: x[0])
        n = len(items)
        # Compute per-trajectory stride from desired val_ratio
        traj_stride = max(2, int(round(1.0 / val_ratio)))
        # Random offset so val frames differ across runs if seed changes
        offset = rng.integers(0, traj_stride)
        for i, (_, ds_idx) in enumerate(items):
            if (i + offset) % traj_stride == 0:
                val_indices.append(ds_idx)
            else:
                test_indices.append(ds_idx)

    print(f"  Interleaved split: {len(val_indices)} val, {len(test_indices)} test "
          f"(ratio {len(val_indices)/(len(val_indices)+len(test_indices)):.1%})")

    return Subset(dataset, val_indices), Subset(dataset, test_indices)


def _contiguous_val_test_split(
    dataset, val_ratio: float = 0.2, seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split held-out data into val/test with contiguous chunks per trajectory.

    Compared with interleaved split, this keeps frame-to-frame action step
    distributions much closer between val and test for BC.
    """
    from torch.utils.data import Subset

    traj_to_indices: Dict[int, List[Tuple[int, int]]] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        t_idx = sample.get("traj_idx", 0)
        f_idx = sample.get("frame_idx", sample.get("timestamp", idx))
        if t_idx not in traj_to_indices:
            traj_to_indices[t_idx] = []
        traj_to_indices[t_idx].append((f_idx, idx))

    rng = np.random.default_rng(seed)
    val_indices: List[int] = []
    test_indices: List[int] = []

    for t_idx in sorted(traj_to_indices.keys()):
        items = sorted(traj_to_indices[t_idx], key=lambda x: x[0])
        ordered_indices = [ds_idx for _, ds_idx in items]
        n = len(ordered_indices)
        if n <= 1:
            test_indices.extend(ordered_indices)
            continue

        n_val = int(round(n * val_ratio))
        n_val = max(1, min(n - 1, n_val))

        # Randomize whether val takes the prefix or suffix to reduce temporal bias.
        if rng.random() < 0.5:
            val_chunk = ordered_indices[:n_val]
            test_chunk = ordered_indices[n_val:]
        else:
            val_chunk = ordered_indices[-n_val:]
            test_chunk = ordered_indices[:-n_val]

        val_indices.extend(val_chunk)
        test_indices.extend(test_chunk)

    total = len(val_indices) + len(test_indices)
    ratio = (len(val_indices) / total) if total > 0 else 0.0
    print(
        f"  Contiguous split: {len(val_indices)} val, {len(test_indices)} test "
        f"(ratio {ratio:.1%})"
    )

    return Subset(dataset, val_indices), Subset(dataset, test_indices)


def create_three_way_split(
    dataset_class,
    root_dir: str,
    test_domains: List,
    val_ratio: float = 0.2,
    seed: int = 42,
    split_mode: str = "contiguous",
    **dataset_kwargs,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train / val / test split with cross-domain isolation.

    Strategy (Scheme A''):
      - Train: all domains NOT in *test_domains* (e.g. Folder 1 + 2)
      - Val + Test: the held-out domain(s) (e.g. Folder 3), further split by
        `split_mode` so val ≈ val_ratio of the held-out data.

    This guarantees:
      • Train ↔ {Val, Test} are fully cross-domain isolated (no leakage).
      • Val and Test share the same domain but use different frames,
        with enough spacing to avoid near-duplicate contamination.
      • Both Val and Test cover all views / anatomies in the held-out domain.

    Args:
        dataset_class: RACINESDataset or UltraBonesDataset
        root_dir: path to dataset root
        test_domains: domains held out for val+test
        val_ratio: fraction of held-out domain for validation (default 0.2)
        seed: random seed for reproducibility
        split_mode: 'contiguous' (default) or 'interleaved'
        **dataset_kwargs: forwarded to dataset constructor

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Step 1: cross-domain split -> train vs held_out
    train_ds, held_out_ds = create_cross_domain_splits(
        dataset_class, root_dir, test_domains, **dataset_kwargs
    )

    # Step 2: split held_out into val and test
    split_mode = str(split_mode).lower()
    if split_mode == "interleaved":
        val_ds, test_ds = _interleaved_val_test_split(
            held_out_ds, val_ratio=val_ratio, seed=seed
        )
    elif split_mode == "contiguous":
        val_ds, test_ds = _contiguous_val_test_split(
            held_out_ds, val_ratio=val_ratio, seed=seed
        )
    else:
        raise ValueError(
            f"Unsupported split_mode: {split_mode}. Expected 'contiguous' or 'interleaved'."
        )

    print(f"Three-way split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Share normalization parameters
    if hasattr(train_ds, "pose_mean"):
        for ds in [val_ds, test_ds]:
            base = ds.dataset if hasattr(ds, "dataset") else ds
            if hasattr(base, "pose_mean"):
                base.pose_mean = train_ds.pose_mean
                base.pose_std = train_ds.pose_std

    return train_ds, val_ds, test_ds


