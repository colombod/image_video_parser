from typing import List, Tuple, Union
from transformers import Owlv2Processor
from transformers.image_transforms import center_to_corners_format
from transformers.models.owlv2.image_processing_owlv2 import box_iou
from torch import TensorType
import torch

class Owlv2ProcessorWithNMS(Owlv2Processor):
    def post_process_object_detection_with_nms(
        self,
        outputs,
        threshold: float = 0.1,
        nms_threshold: float = 0.3,
        target_sizes: Union[TensorType, List[Tuple]] = None,
    ):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            nms_threshold (`float`, *optional*):
                IoU threshold to filter overlapping objects the raw detections.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

        # Apply non-maximum suppression (NMS)
        # borrowed the implementation from HuggingFace Owlv2 post_process_image_guided_detection()
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlv2/image_processing_owlv2.py#L563-L573
        if nms_threshold < 1.0:
            for idx in range(boxes.shape[0]):
                for i in torch.argsort(-scores[idx]):
                    if not scores[idx][i]:
                        continue
                    ious = box_iou(boxes[idx][i, :].unsqueeze(0), boxes[idx])[0][0]
                    ious[i] = -1.0  # Mask self-IoU.
                    scores[idx][ious > nms_threshold] = 0.0

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            # rescale coordinates
            width_ratio = 1
            height_ratio = 1

            if img_w < img_h:
                width_ratio = img_w / img_h
            elif img_h < img_w:
                height_ratio = img_h / img_w

            img_w = img_w / width_ratio
            img_h = img_h / height_ratio

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(
                boxes.device
            )
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results