import os.path as osp

import numpy as np
import torch

from .custom import CustomDataset


class ScanNetDataset(CustomDataset):

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
    BENCHMARK_SEMANTIC_IDXS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

    def load(self, filename):
        scan_id = osp.basename(filename).replace(self.suffix, "")
        ps_filename = osp.join(self.data_root, self.label_type, scan_id + ".pth")

        if self.prefix == "test":
            xyz, rgb = torch.load(filename)
            semantic_label = np.zeros(xyz.shape[0], dtype=np.long)
            instance_label = np.zeros(xyz.shape[0], dtype=np.long)
            prob_label = np.zeros(xyz.shape[0], dtype=np.float)
            mu_label = np.zeros(xyz.shape[0], dtype=np.float)
            var_label = np.zeros(xyz.shape[0], dtype=np.float)
        else:
            xyz, rgb, _, _ = torch.load(filename)
            semantic_label, instance_label, prob_label, mu_label, var_label = torch.load(
                ps_filename
            )  # NOTE best label

        spp_filename = osp.join(self.data_root, "superpoints", scan_id + ".pth")
        spp = torch.load(spp_filename)

        return xyz, rgb, semantic_label, instance_label, prob_label, mu_label, var_label, spp
