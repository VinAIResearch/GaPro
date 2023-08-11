import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from isbnet.model.matcher import HungarianMatcher

from .model_utils import batch_giou_corres, giou_aabb, is_within_bb_torch


@torch.no_grad()
def get_iou(inputs, targets, thresh=0.5):
    inputs_bool = inputs.detach().sigmoid()
    inputs_bool = inputs_bool >= thresh

    intersection = (inputs_bool * targets).sum(-1)
    union = inputs_bool.sum(-1) + targets.sum(-1) - intersection

    iou = intersection / (union + 1e-6)

    return iou


def compute_dice_loss(inputs, targets, num_boxes, mask=None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()

    if mask is not None:
        inputs = inputs * mask
        targets = targets * mask

    # inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / (num_boxes + 1e-6)


def compute_sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, mask=None):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if mask is not None:
        ce_loss = ce_loss * mask

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / (num_boxes + 1e-6)


class Criterion(nn.Module):
    def __init__(
        self,
        instance_classes=18,
        semantic_weight=None,
        ignore_label=-100,
        eos_coef=0.1,
        semantic_only=True,
        total_epoch=40,
        trainall=False,
        voxel_scale=50,
    ):
        super(Criterion, self).__init__()

        self.matcher = HungarianMatcher()
        self.semantic_only = semantic_only

        self.ignore_label = ignore_label

        self.label_shift = 1
        self.semantic_classes = instance_classes + 1
        self.instance_classes = instance_classes
        self.semantic_weight = semantic_weight
        self.eos_coef = eos_coef
        self.voxel_scale = voxel_scale

        self.total_epoch = total_epoch

        self.trainall = trainall

        empty_weight = torch.ones(self.instance_classes + 1)
        empty_weight[-1] = self.eos_coef
        if self.semantic_weight:
            for i in range(self.label_shift, self.instance_classes + self.label_shift):
                empty_weight[i - self.label_shift] = self.semantic_weight[i]

        self.register_buffer("empty_weight", empty_weight)

        # self.loss_weight = {
        #     "dice_loss": 4,
        #     # "focal_loss": 4,
        #     "bce_loss": 4,
        #     "cls_loss": 1,
        #     "iou_loss": 1,
        #     "box_loss": 1,
        #     "giou_loss": 1,
        # }

        self.loss_weight = {
            "dice_loss": 1,
            "bce_loss": 1,
            "cls_loss": 0.5,
            "iou_loss": 0.5,
            "box_loss": 0.5,
            "giou_loss": 0.5,
            "levelset_loss": 0.5,
            "kl_loss": 0.1,
        }

    def cal_point_wise_loss(
        self,
        semantic_scores,
        # centroid_offset,
        corners_offset,
        box_conf,
        semantic_labels,
        instance_labels,
        # centroid_offset_labels,
        corners_offset_labels,
        coords_float,
    ):
        losses = {}

        if self.semantic_weight:
            weight = torch.tensor(self.semantic_weight, dtype=torch.float, device=semantic_labels.device)
        else:
            weight = None
        semantic_loss = F.cross_entropy(
            semantic_scores, semantic_labels, ignore_index=self.ignore_label, weight=weight
        )

        losses["pw_sem_loss"] = semantic_loss

        pos_inds = instance_labels != self.ignore_label
        total_pos_inds = pos_inds.sum()
        if total_pos_inds == 0:
            # offset_loss = 0 * centroid_offset.sum()
            offset_vertices_loss = 0 * corners_offset.sum()
            conf_loss = 0 * box_conf.sum()
            giou_loss = 0 * box_conf.sum()
        else:
            # offset_loss = (
            #     F.l1_loss(centroid_offset[pos_inds], centroid_offset_labels[pos_inds], reduction="sum")
            #     / total_pos_inds
            # )

            offset_vertices_loss = (
                F.l1_loss(corners_offset[pos_inds], corners_offset_labels[pos_inds], reduction="sum") / total_pos_inds
            )

            iou_gt, giou = batch_giou_corres(
                corners_offset[pos_inds] + coords_float[pos_inds].repeat(1, 2),
                corners_offset_labels[pos_inds] + coords_float[pos_inds].repeat(1, 2),
            )
            giou_loss = torch.sum((1 - giou)) / total_pos_inds

            iou_gt = iou_gt.detach()
            conf_loss = F.mse_loss(box_conf[pos_inds], iou_gt, reduction="sum") / total_pos_inds

        # losses["pw_center_loss"] = offset_loss * self.voxel_scale / 50.0
        losses["pw_corners_loss"] = offset_vertices_loss * self.voxel_scale / 50.0
        losses["pw_giou_loss"] = giou_loss
        losses["pw_conf_loss"] = conf_loss

        return losses

    def levelset_loss(self, levelset_coords_, levelset_feats_, mask_logit_pred, box_label):
        num_gts = len(box_label)

        is_within_conds = is_within_bb_torch(
            levelset_coords_[None, :, :], box_label[:, None, :3] - 0.005, box_label[:, None, 3:] + 0.005
        )  # N_box, N_points

        min_points_conds = torch.sum(is_within_conds, dim=1) > 0

        is_within_conds_ = is_within_conds[min_points_conds]  # N_box_, N_points
        mask_logit_pred_ = mask_logit_pred[min_points_conds]

        box_inds, point_inds = torch.nonzero((is_within_conds_), as_tuple=True)

        mask_logit_pred_sm_insts = torch.sigmoid(mask_logit_pred_[box_inds, point_inds])  # M
        levelset_feats_insts = levelset_feats_[point_inds, :]  # M, f

        _, box_inds_norm = torch.unique(box_inds, return_inverse=True)

        ave_similarity = (
            torch_scatter.scatter(
                (levelset_feats_insts * mask_logit_pred_sm_insts[:, None]).float(),
                index=box_inds_norm,
                dim=0,
                reduce="sum",
            )
            / torch_scatter.scatter(mask_logit_pred_sm_insts.float(), index=box_inds_norm, reduce="sum").clamp(
                min=0.00001
            )[:, None]
        )

        region_level = levelset_feats_insts - ave_similarity[box_inds_norm, :]

        region_level_loss = region_level * region_level * mask_logit_pred_sm_insts[:, None]
        region_level_loss = torch_scatter.scatter(
            region_level_loss.sum(dim=1).float(), index=box_inds_norm, reduce="mean"
        )  # n_box

        region_level_loss = region_level_loss.sum() / (num_gts + 1e-4)

        return region_level_loss

    def single_layer_loss(
        self,
        cls_logits,
        mask_logits_list,
        conf_logits,
        box_preds,
        prob_labels,
        batch_offset,
        levelset_feats,
        levelset_coords,
        row_indices,
        cls_labels,
        inst_labels,
        box_labels,
        batch_size,
    ):
        loss_dict = {}

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True, device=cls_logits.device, dtype=torch.float)

        num_gt = 0
        for b in range(batch_size):
            b_s, b_e = batch_offset[b], batch_offset[b + 1]
            mask_logit_b = mask_logits_list[b]
            cls_logit_b = cls_logits[b]  # n_queries x n_classes
            conf_logits_b = conf_logits[b]  # n_queries
            box_preds_b = box_preds[b]

            prob_labels_b = prob_labels[b_s:b_e]

            pred_inds, cls_label, inst_label, box_label = row_indices[b], cls_labels[b], inst_labels[b], box_labels[b]

            n_queries = cls_logit_b.shape[0]

            if mask_logit_b is None:
                continue

            if pred_inds is None:
                continue

            mask_logit_pred = mask_logit_b[pred_inds]
            conf_logits_pred = conf_logits_b[pred_inds]
            box_pred = box_preds_b[pred_inds]

            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch

            loss_dict["dice_loss"] = loss_dict["dice_loss"] + compute_dice_loss(
                mask_logit_pred, inst_label, num_gt_batch
            )

            bce_loss = F.binary_cross_entropy_with_logits(mask_logit_pred, inst_label, reduction="none")
            bce_loss = ((bce_loss * prob_labels_b).sum() / prob_labels_b.sum()).sum() / (num_gt_batch + 1e-6)
            # bce_loss = bce_loss.mean(1).sum() / (num_gt_batch + 1e-6)

            loss_dict["bce_loss"] = loss_dict["bce_loss"] + bce_loss

            gt_iou = get_iou(mask_logit_pred, inst_label)

            loss_dict["iou_loss"] = (
                loss_dict["iou_loss"] + F.mse_loss(conf_logits_pred, gt_iou, reduction="sum") / num_gt_batch
            )

            target_classes = (
                torch.ones((n_queries), dtype=torch.int64, device=cls_logits.device) * self.instance_classes
            )

            target_classes[pred_inds] = cls_label

            loss_dict["cls_loss"] = loss_dict["cls_loss"] + F.cross_entropy(
                cls_logit_b,
                target_classes,
                self.empty_weight,
                reduction="mean",
            )

            loss_dict["box_loss"] = (
                loss_dict["box_loss"]
                + (self.voxel_scale / 50.0) * F.l1_loss(box_pred, box_label, reduction="sum") / num_gt_batch
            )

            iou_gt, giou = giou_aabb(box_pred, box_label, coords=None)

            loss_dict["giou_loss"] = loss_dict["giou_loss"] + torch.sum(1 - giou) / num_gt_batch

            # NOTE level_set loss
            levelset_coords_ = levelset_coords[b_s:b_e, :]
            levelset_feats_ = levelset_feats[b_s:b_e, :]

            region_level_loss = self.levelset_loss(levelset_coords_, levelset_feats_, mask_logit_pred, box_label)
            loss_dict["levelset_loss"] = loss_dict["levelset_loss"] + region_level_loss

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] / batch_size

        return loss_dict

    def forward(self, batch_inputs, model_outputs):
        loss_dict = {}

        semantic_labels = batch_inputs["semantic_labels"]
        instance_labels = batch_inputs["instance_labels"]

        if model_outputs is None:
            loss_dict["Placeholder"] = torch.tensor(
                0.0, requires_grad=True, device=semantic_labels.device, dtype=torch.float
            )

            return loss_dict

        if self.semantic_only or self.trainall:
            # '''semantic loss'''
            semantic_scores = model_outputs["semantic_scores"]
            # centroid_offset = model_outputs["centroid_offset"]
            corners_offset = model_outputs["corners_offset"]
            box_conf = model_outputs["box_conf"]

            coords_float = batch_inputs["coords_float"]
            # centroid_offset_labels = batch_inputs["centroid_offset_labels"]
            corners_offset_labels = batch_inputs["corners_offset_labels"]

            point_wise_loss = self.cal_point_wise_loss(
                semantic_scores,
                # centroid_offset,
                corners_offset,
                box_conf,
                semantic_labels,
                instance_labels,
                # centroid_offset_labels,
                corners_offset_labels,
                coords_float,
            )

            loss_dict.update(point_wise_loss)

            if self.semantic_only:
                return loss_dict

        for k in loss_dict.keys():
            if "pw" in k:
                loss_dict[k] = loss_dict[k] * 0.25

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True, device=semantic_labels.device, dtype=torch.float)

        """ Main loss """
        cls_logits = model_outputs["cls_logits"]
        mask_logits = model_outputs["mask_logits"]
        conf_logits = model_outputs["conf_logits"]
        box_preds = model_outputs["box_preds"]

        dc_inst_mask_arr = model_outputs["dc_inst_mask_arr"]
        dc_prob_labels = model_outputs["dc_prob_labels"]
        dc_mu_labels = model_outputs["dc_mu_labels"]
        dc_var_labels = model_outputs["dc_var_labels"]
        dc_batch_offsets = model_outputs["dc_batch_offsets"]

        dc_rgb_feats = model_outputs["dc_rgb_feats"]
        dc_coords_float = model_outputs["dc_coords_float"]

        batch_size, n_queries = cls_logits.shape[:2]

        gt_dict, _ = self.matcher.forward_dup(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            dc_inst_mask_arr,
            dup_gt=4,
        )

        # NOTE main loss

        row_indices = gt_dict["row_indices"]
        inst_labels = gt_dict["inst_labels"]
        cls_labels = gt_dict["cls_labels"]
        box_labels = gt_dict["box_labels"]

        main_loss_dict = self.single_layer_loss(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            dc_prob_labels,
            dc_batch_offsets,
            dc_rgb_feats,
            dc_coords_float,
            row_indices,
            cls_labels,
            inst_labels,
            box_labels,
            batch_size,
        )

        for k, v in self.loss_weight.items():
            loss_dict[k] = loss_dict[k] + main_loss_dict[k] * v

        # NOTE aux loss

        mu_labels = model_outputs["dc_mu_labels"]
        var_labels = model_outputs["dc_var_labels"]
        mu_pred = model_outputs["mu_pred"]
        logvar_pred = model_outputs["logvar_pred"]

        loss_dict["kl_loss"] = torch.tensor(0.0, requires_grad=True, device=instance_labels.device, dtype=torch.float)

        epsilon = 1e-4
        mask_kl_varzero = (mu_labels != -100) & (var_labels != -100) & (var_labels <= epsilon)
        mask_kl_var = (mu_labels != -100) & (var_labels != -100) & (var_labels > epsilon)

        if mask_kl_varzero.sum() > 0:
            loss_kl_varzero = (torch.exp(logvar_pred[mask_kl_varzero]) - 1) ** 2 + (
                mu_pred[mask_kl_varzero] - mu_labels[mask_kl_varzero]
            ) ** 2
            loss_kl_varzero = loss_kl_varzero.sum() / (mask_kl_varzero.sum() + 1e-4)

            loss_dict["kl_loss"] = loss_dict["kl_loss"] + loss_kl_varzero * self.loss_weight["kl_loss"]

        if mask_kl_var.sum() > 0:
            loss_kl_var = (
                (logvar_pred[mask_kl_var] - torch.log(var_labels[mask_kl_var]))
                + ((mu_pred[mask_kl_var] - mu_labels[mask_kl_var]) ** 2 + var_labels[mask_kl_var] ** 2)
                * (torch.exp(-2 * logvar_pred[mask_kl_var]))
                - 0.5
            )
            loss_kl_var = loss_kl_var.sum() / (mask_kl_var.sum() + 1e-4)

            loss_dict["kl_loss"] = loss_dict["kl_loss"] + loss_kl_var * self.loss_weight["kl_loss"]

        return loss_dict
