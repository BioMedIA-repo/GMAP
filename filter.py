import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Any

from torch import Tensor


def sigmoid_rampup(current_iter, rampup_length):
    if rampup_length == 0:
        return 1.0
    if current_iter >= rampup_length:
        return 1.0

    current = max(0.0, float(current_iter))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))



class EpochProxySelector:
    def __init__(self, num_classes, args=None):
        self.score_dict = {}
        self.topK_ratio = 0.0
        self.num_classes = num_classes
        self.log_c = torch.log(torch.tensor(self.num_classes, dtype=torch.float32))
        self.args = args
        self.score_threshold = 0.3
        self.last_epoch_threshold = 0.0

    def reset_epoch(self):
        self.last_epoch_threshold = self.score_threshold
        self.score_dict.clear()

        self.score_threshold = self.last_epoch_threshold

    def update_topk_ratio(self, iter_num, rampup_length=40000):
        start_k = self.args.k if self.args and hasattr(self.args, 'k') else 0.1
        ramp = sigmoid_rampup(iter_num, rampup_length)
        self.topK_ratio = start_k + (1.0 - start_k) * ramp
        self.topK_ratio = min(self.topK_ratio, 1.0)

    def _compute_pixel_weights(self, predictions_list: List[torch.Tensor]) -> Tensor:

        device = predictions_list[0].device
        self.log_c = self.log_c.to(device)

        stacked_preds = torch.stack(predictions_list, dim=1)
        epsilon = 1e-6

        mean_pred = torch.mean(stacked_preds, dim=1)
        entropy_of_mean = -torch.sum(mean_pred * torch.log(mean_pred + epsilon), dim=1)

        entropies = -torch.sum(stacked_preds * torch.log(stacked_preds + epsilon), dim=2)
        mean_of_entropies = torch.mean(entropies, dim=1)

        jsd_pixels = entropy_of_mean - mean_of_entropies

        pixel_consistency = 1 - (jsd_pixels / self.log_c)
        pixel_certainty = 1 - (entropy_of_mean / self.log_c)

        pixel_weights = (pixel_consistency * pixel_certainty).detach()
        return pixel_weights  # Shape: (B, H, W)

    def _compute_sample_scores(self, predictions_list: List[torch.Tensor]) -> Tensor:
        pixel_weights = self._compute_pixel_weights(predictions_list)
        sample_scores = torch.mean(pixel_weights, dim=(1, 2))

        mean_pred = torch.mean(torch.stack(predictions_list, dim=0), dim=0)
        pseudo_labels = torch.argmax(mean_pred, dim=1)
        has_foreground = torch.any((pseudo_labels > 0).flatten(start_dim=1), dim=1)
        sample_scores[~has_foreground] = -1.0

        return sample_scores, pixel_weights

    def update_score_distribution(self, predictions_list: List[torch.Tensor]):
        sample_scores, _ = self._compute_sample_scores(predictions_list)

        for score in sample_scores.cpu():
            score = float(score)
            if score >= 0:  
                self.score_dict[len(self.score_dict)] = score

    def update_selection_threshold(self, global_step=50):
        if not self.score_dict or len(self.score_dict) < global_step:
            return

        all_scores = list(self.score_dict.values())
        all_scores.sort(reverse=True)

        topk_index = int(len(all_scores) * self.topK_ratio)
        topk_index = min(topk_index, len(all_scores) - 1)

        if topk_index >= 0:
            self.score_threshold = all_scores[topk_index]

    def filter_batch(
            self,
            predictions_list: List[torch.Tensor],
            *other_tensors: Tensor
    ) -> Tuple[List[torch.Tensor], Tensor, Tuple[Tensor, ...]]:
        device = predictions_list[0].device
        batch_size = predictions_list[0].size(0)

        sample_scores, pixel_weights = self._compute_sample_scores(predictions_list)

        selected_mask = (sample_scores >= self.score_threshold) & (sample_scores >= 0)
        filtered_preds_list = [p[selected_mask] for p in predictions_list]
        filtered_pixel_weights = pixel_weights[selected_mask]
        filtered_other_tensors = tuple(t[selected_mask] for t in other_tensors)

        expected_num = max(1, int(batch_size * self.topK_ratio))

        if filtered_preds_list[0].size(0) >= expected_num:
            return filtered_preds_list, filtered_pixel_weights, filtered_other_tensors
        else:
            num_needed = expected_num - filtered_preds_list[0].size(0)
            remaining_mask = ~selected_mask
            if remaining_mask.sum() == 0:
                return filtered_preds_list, filtered_pixel_weights, filtered_other_tensors
            remaining_preds_list = [p[remaining_mask] for p in predictions_list]
            remaining_pixel_weights = pixel_weights[remaining_mask]
            remaining_other_tensors = tuple(t[remaining_mask] for t in other_tensors)
            remaining_scores = sample_scores[remaining_mask]
            num_to_select = min(num_needed, remaining_scores.size(0))

            if remaining_scores.size(0) > 0:
                _, topk_indices_in_remaining = torch.topk(remaining_scores, k=num_to_select)
            else: 
                topk_indices_in_remaining = torch.tensor([], dtype=torch.long, device=device)

            if topk_indices_in_remaining.numel() > 0:
                supplement_preds_list = [p[topk_indices_in_remaining] for p in remaining_preds_list]
                supplement_pixel_weights = remaining_pixel_weights[topk_indices_in_remaining]
                supplement_other_tensors = tuple(t[topk_indices_in_remaining] for t in remaining_other_tensors)

                final_preds_list = [torch.cat([p_filt, p_supp], dim=0) for p_filt, p_supp in
                                    zip(filtered_preds_list, supplement_preds_list)]
                final_pixel_weights = torch.cat([filtered_pixel_weights, supplement_pixel_weights], dim=0)
                final_other_tensors = tuple(torch.cat([t_filt, t_supp], dim=0) for t_filt, t_supp in
                                            zip(filtered_other_tensors, supplement_other_tensors))
            else:
                final_preds_list = filtered_preds_list
                final_pixel_weights = filtered_pixel_weights
                final_other_tensors = filtered_other_tensors

            return final_preds_list, final_pixel_weights, final_other_tensors
