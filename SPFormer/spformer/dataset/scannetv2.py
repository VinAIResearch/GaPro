import glob
import math
import os.path as osp
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pointgroup_ops
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset

from ..utils import Instances3D


class ScanNetDataset(Dataset):

    CLASSES = (
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refrigerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    )
    NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
    BENCHMARK_SEMANTIC_IDXS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

    def __init__(
        self,
        data_root,
        prefix,
        suffix,
        voxel_cfg=None,
        training=True,
        with_label=True,
        mode=4,
        with_elastic=True,
        use_xyz=True,
        logger=None,
        repeat=1,
        label_type='default',
    ):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.mode = mode
        self.with_elastic = with_elastic
        self.use_xyz = use_xyz
        self.logger = logger
        self.repeat = repeat
        self.label_type = label_type
        self.filenames = self.get_filenames()
        self.logger.info(f"Load {self.prefix} dataset: {len(self.filenames)} scans")

    def load(self, filename):
        if self.with_label:
            return torch.load(filename)
        else:
            xyz, rgb = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, dummy_sem_label, dummy_inst_label

    def get_filenames(self):
        if self.prefix == "trainval":
            filenames_train = glob.glob(osp.join(self.data_root, "train", "*" + self.suffix))
            filenames_val = glob.glob(osp.join(self.data_root, "val", "*" + self.suffix))
            filenames = filenames_train + filenames_val
        else:
            filenames = glob.glob(osp.join(self.data_root, self.prefix, "*" + self.suffix))
        assert len(filenames) > 0, "Empty dataset."
        filenames = sorted(filenames * self.repeat)

        # filenames = filenames[:30]

        return filenames

        # filenames = glob.glob(osp.join(self.data_root, self.prefix, '*' + self.suffix))
        # assert len(filenames) > 0, 'Empty dataset.'
        # filenames = sorted(filenames * self.repeat)
        # # filenames = filenames[:12]
        # return filenames

    def __len__(self):
        return len(self.filenames)

    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label, prob_label, mu_label, var_label):
        xyz_middle = self.data_aug(xyz, True, True, True)
        rgb += np.random.randn(3) * 0.1
        xyz = xyz_middle * self.voxel_cfg.scale
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.0)
            xyz = self.elastic(xyz, 20, 160.0)
        xyz = xyz - xyz.min(0)
        xyz, valid_idxs = self.crop(xyz)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]

        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        prob_label = prob_label[valid_idxs]
        mu_label = mu_label[valid_idxs]
        var_label = var_label[valid_idxs]

        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, prob_label, mu_label, var_label

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label, prob_label, mu_label, var_label):
        xyz_middle = xyz
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, prob_label, mu_label, var_label

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
            )  # rotation
        return np.matmul(xyz, m)

    def crop(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        r"""
        crop the point cloud to reduce training complexity

        Args:
            xyz (np.ndarray, [N, 3]): input point cloud to be cropped

        Returns:
            Union[np.ndarray, np.ndarray]: processed point cloud and boolean valid indices
        """
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.voxel_cfg.max_npoint:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def elastic(self, xyz, gran, mag):
        """Elastic distortion (from point group)

        Args:
            xyz (np.ndarray): input point cloud
            gran (float): distortion param
            mag (float): distortion scalar

        Returns:
            xyz: point cloud with elastic distortion
        """
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag

    def get_cropped_inst_label(self, instance_label: np.ndarray, valid_idxs: np.ndarray) -> np.ndarray:
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def get_instance3D(self, instance_label, semantic_label, coord_float, superpoint, scan_id):
        num_insts = instance_label.max().item() + 1
        num_points = len(instance_label)
        gt_masks, gt_labels, gt_boxes = [], [], []

        gt_inst = torch.zeros(num_points, dtype=torch.int64)
        for i in range(num_insts):
            idx = torch.where(instance_label == i)
            assert len(torch.unique(semantic_label[idx])) == 1
            sem_id = semantic_label[idx][0]
            if sem_id == -100:
                # sem_id = 1
                # gt_inst[idx] = sem_id * 1000 + i + 1
                continue
            gt_mask = torch.zeros(num_points)
            gt_mask[idx] = 1
            gt_masks.append(gt_mask)
            gt_label = sem_id
            gt_labels.append(gt_label)
            gt_inst[idx] = (sem_id + 1) * 1000 + i + 1

            coord_float_ = coord_float[idx]

            min_coord_float_ = coord_float_.min(0)[0]
            max_coord_float_ = coord_float_.max(0)[0]
            # if sem_label == -100: continue
            gt_boxes.append(torch.cat([min_coord_float_, max_coord_float_], dim=0))

        if gt_masks:
            gt_masks = torch.stack(gt_masks, dim=0)
            gt_spmasks = torch_scatter.scatter_mean(gt_masks.float(), superpoint, dim=-1)
            gt_spmasks = (gt_spmasks > 0.5).float()
            gt_boxes = torch.stack(gt_boxes)

        else:
            gt_spmasks = torch.tensor([])
            gt_boxes = torch.tensor([])
        gt_labels = torch.tensor(gt_labels)

        inst = Instances3D(num_points, gt_instances=gt_inst.numpy())
        inst.gt_labels = gt_labels.long()
        inst.gt_spmasks = gt_spmasks
        inst.gt_boxes = gt_boxes
        return inst

    def __getitem__(self, index: int) -> Tuple:
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, "")
        spp_filename = osp.join(self.data_root, "superpoints", scan_id + ".pth")
        ps_filename = osp.join(self.data_root, self.label_type, scan_id + ".pth")

        if self.prefix != 'test':
            xyz, rgb, _, _ = torch.load(filename)
            semantic_label, instance_label, prob_label, mu_label, var_label = torch.load(
                ps_filename
            )
        else:
            xyz, rgb = torch.load(filename)

            semantic_label = np.zeros((xyz.shape[0]), dtype=np.long)
            instance_label = np.zeros((xyz.shape[0]), dtype=np.long)
            prob_label = np.zeros(xyz.shape[0], dtype=np.float)
            mu_label = np.zeros(xyz.shape[0], dtype=np.float)
            var_label = np.zeros(xyz.shape[0], dtype=np.float)

        assert xyz.shape[0] == semantic_label.shape[0]
        superpoint = torch.load(spp_filename).numpy()

        xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, prob_label, mu_label, var_label = (
            self.transform_train(xyz, rgb, superpoint, semantic_label, instance_label, prob_label, mu_label, var_label)
            if self.training
            else self.transform_test(
                xyz, rgb, superpoint, semantic_label, instance_label, prob_label, mu_label, var_label
            )
        )

        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoint)
        semantic_label = torch.from_numpy(semantic_label).long()

        semantic_label = torch.where(semantic_label == 18, -100, semantic_label)

        instance_label = torch.from_numpy(instance_label).long()

        prob_label = torch.from_numpy(prob_label).float()
        mu_label = torch.from_numpy(mu_label).float()
        var_label = torch.from_numpy(var_label).float()

        inst = self.get_instance3D(instance_label, semantic_label, coord_float, superpoint, scan_id)
        return scan_id, coord, coord_float, feat, superpoint, prob_label, mu_label, var_label, inst

    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        scan_ids, coords, coords_float, feats, superpoints, prob_labels, mu_labels, var_labels, insts = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        batch_offsets = [0]
        superpoint_bias = 0

        for i, data in enumerate(batch):
            scan_id, coord, coord_float, feat, superpoint, prob_label, mu_label, var_label, inst = data

            superpoint += superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            batch_offsets.append(superpoint_bias)

            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            prob_labels.append(prob_label)
            mu_labels.append(mu_label)
            var_labels.append(var_label)
            insts.append(inst)

        # merge all scan in batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        prob_labels = torch.cat(prob_labels, 0).float()
        mu_labels = torch.cat(mu_labels, 0).float()
        var_labels = torch.cat(var_labels, 0).float()

        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)

        # voxelize
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0], None)  # long [3]
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords, len(batch), self.mode)

        return {
            "scan_ids": scan_ids,
            "voxel_coords": voxel_coords,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "spatial_shape": spatial_shape,
            "feats": feats,
            "coords_float": coords_float,
            "superpoints": superpoints,
            "prob_labels": prob_labels,
            "mu_labels": mu_labels,
            "var_labels": var_labels,
            "batch_offsets": batch_offsets,
            "insts": insts,
        }
