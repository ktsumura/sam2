import torch

from training.loss_fns import MultiStepMultiMasksAndIous


class MultiStepMultiMasksAndIousIfTargetIsPresent(MultiStepMultiMasksAndIous):
    def __init__(
            self,
            weight_dict,
            focal_alpha=0.25,
            focal_gamma=2,
            supervise_all_iou=False,
            iou_use_l1_loss=False,
            pred_obj_scores=False,
            focal_gamma_obj_score=0.0,
            focal_alpha_obj_score=-1,
            batch_size=1,
    ):
        super().__init__(
            weight_dict,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            supervise_all_iou=supervise_all_iou,
            iou_use_l1_loss=iou_use_l1_loss,
            pred_obj_scores=pred_obj_scores,
            focal_gamma_obj_score=focal_gamma_obj_score,
            focal_alpha_obj_score=focal_alpha_obj_score,
        )

    def _update_losses(
            self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        num_objects = num_objects / target_masks.shape[0]

        for src_mask, target_mask, iou, object_score_logit in zip(
                src_masks, target_masks, ious, object_score_logits
        ):
            src_mask = torch.unsqueeze(src_mask, axis=0)
            target_mask = torch.unsqueeze(target_mask, axis=0)
            iou = torch.unsqueeze(iou, axis=0)
            object_score_logit = torch.unsqueeze(object_score_logit, axis=0)

            # If there is no target object, skip all loss calculation in the same way as the Neo segmentation training.
            # It is unknown whether a target object is really not present or ground truth is not available
            # without the label integrity stored in HDF files.
            # Note super()._update_losses calculates loss_class when a target object is not present.
            if torch.any(target_mask > 0):
                super()._update_losses(
                    losses, src_mask, target_mask, iou, num_objects, object_score_logit
                )
