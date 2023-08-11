import functools

import gorilla
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from spformer.utils import cuda_cast, rle_encode, rle_encode_gpu_batch
from torch_scatter import scatter_max, scatter_mean

from .backbone import MLP, ResidualBlock, UBlock
from .loss import Criterion
from .query_decoder import QueryDecoder


@gorilla.MODELS.register_module()
class SPFormer(nn.Module):
    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool="mean",
        num_class=18,
        decoder=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
    ):
        super().__init__()

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool
        self.num_class = num_class

        # self.uncertainty_linear = MLP(media, 1, norm_fn=norm_fn, num_layers=2)
        self.mu_linear = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
        self.logvar_linear = MLP(media, 1, norm_fn=norm_fn, num_layers=3)

        # decoder
        self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)

        # criterion
        self.criterion = Criterion(**criterion, num_class=num_class)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(SPFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode="loss"):
        if mode == "loss":
            return self.loss(**batch)
        elif mode == "predict":
            return self.predict(**batch)

    @cuda_cast
    def loss(
        self,
        scan_ids,
        voxel_coords,
        p2v_map,
        v2p_map,
        spatial_shape,
        feats,
        coords_float,
        insts,
        superpoints,
        prob_labels,
        mu_labels,
        var_labels,
        batch_offsets,
        **kwargs
    ):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        voxel_rgb_feats = voxel_feats[:, :3].clone()

        voxel_coords_float = pointgroup_ops.voxelization(coords_float, v2p_map)
        sp_feats, sp_coords, sp_rgb_feats, sp_prob_labels, sp_mu_labels, sp_var_labels = self.extract_feat(
            input,
            voxel_coords_float,
            voxel_rgb_feats,
            superpoints,
            p2v_map,
            prob_labels=prob_labels,
            mu_labels=mu_labels,
            var_labels=var_labels,
        )

        sp_mu_preds = self.mu_linear(sp_feats).squeeze(-1)
        sp_logvar_preds = self.logvar_linear(sp_feats).squeeze(-1)

        out = self.decoder(sp_feats, batch_offsets)

        out["sp_coords"] = sp_coords
        out["sp_rgb_feats"] = sp_rgb_feats
        out["batch_offsets"] = batch_offsets
        out["sp_prob_labels"] = sp_prob_labels
        out["sp_mu_labels"] = sp_mu_labels
        out["sp_var_labels"] = sp_var_labels
        out["sp_mu_preds"] = sp_mu_preds
        out["sp_logvar_preds"] = sp_logvar_preds

        loss, loss_dict = self.criterion(out, insts)
        return loss, loss_dict

    @cuda_cast
    def predict(
        self,
        scan_ids,
        voxel_coords,
        p2v_map,
        v2p_map,
        spatial_shape,
        feats,
        coords_float,
        insts,
        superpoints,
        prob_labels,
        batch_offsets,
        **kwargs
    ):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)

        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        voxel_rgb_feats = voxel_feats[:, :3].clone()

        voxel_coords_float = pointgroup_ops.voxelization(coords_float, v2p_map)
        sp_feats, sp_coords, _ = self.extract_feat(input, voxel_coords_float, voxel_rgb_feats, superpoints, p2v_map)

        out = self.decoder(sp_feats, batch_offsets)

        ret = self.predict_by_feat(scan_ids, out, superpoints, insts, coords_float)
        return ret

    def predict_by_feat(self, scan_ids, out, superpoints, insts, coords_float):
        pred_labels = out["labels"]
        pred_masks = out["masks"]
        pred_scores = out["scores"]

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]
        labels = (
            torch.arange(self.num_class, device=scores.device)
            .unsqueeze(0)
            .repeat(self.decoder.num_query, 1)
            .flatten(0, 1)
        )
        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]
        labels += 1

        topk_idx = torch.div(topk_idx, self.num_class, rounding_mode="floor")
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        # mask_pred before sigmoid()
        mask_pred = (mask_pred > 0).float()  # [n_p, M]
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores
        # get mask
        mask_pred = mask_pred[:, superpoints].int()

        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]  # (n_p,)
        labels = labels[score_mask]  # (n_p,)
        mask_pred = mask_pred[score_mask]  # (n_p, N)

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]  # (n_p,)
        labels = labels[npoint_mask]  # (n_p,)
        mask_pred = mask_pred[npoint_mask]  # (n_p, N)

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred_rle = rle_encode_gpu_batch(mask_pred)

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred["scan_id"] = scan_ids[0]
            pred["label_id"] = cls_pred[i]
            pred["conf"] = score_pred[i]
            # rle encode mask to save memory
            pred["pred_mask"] = mask_pred_rle[i]

            mask_ = mask_pred[i]
            coords_ = coords_float[mask_.bool(), :]
            box = torch.cat([coords_.min(dim=0)[0], coords_.max(dim=0)[0]]).cpu().numpy()
            pred["box"] = box

            pred_instances.append(pred)

        gt_instances = insts[0].gt_instances
        return dict(scan_id=scan_ids[0], pred_instances=pred_instances, gt_instances=gt_instances)

    def extract_feat(
        self, x, coords_float, rgb_feats, superpoints, v2p_map, prob_labels=None, mu_labels=None, var_labels=None
    ):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)

        coords_float = coords_float[v2p_map.long()]
        rgb_feats = rgb_feats[v2p_map.long()]

        # superpoint pooling
        if self.pool == "mean":
            x = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
            coords_float_p = scatter_mean(coords_float, superpoints, dim=0)
            rgb_feats_p = scatter_mean(rgb_feats, superpoints, dim=0)

            if prob_labels is not None:
                prob_labels_p = scatter_mean(prob_labels, superpoints, dim=0)
                mu_labels_p = scatter_mean(mu_labels, superpoints, dim=0)
                var_labels_p = scatter_mean(var_labels, superpoints, dim=0)

        elif self.pool == "max":
            x, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)
            coords_float_p, _ = scatter_max(coords_float, superpoints, dim=0)
            rgb_feats_p, _ = scatter_max(rgb_feats, superpoints, dim=0)

            if prob_labels is not None:
                prob_labels_p, _ = scatter_max(prob_labels, superpoints, dim=0)
                mu_labels_p, _ = scatter_max(mu_labels, superpoints, dim=0)
                var_labels_p, _ = scatter_max(var_labels, superpoints, dim=0)

        if prob_labels is not None:
            return x, coords_float_p, rgb_feats_p, prob_labels_p, mu_labels_p, var_labels_p

        return x, coords_float_p, rgb_feats_p
