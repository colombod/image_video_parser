import abc
from typing import Optional
import torch
import numpy as np
from PIL import Image

from .utils import image_to_base64, try_get_source_ref_node_info, ImageRegion
from sam2.sam2_image_predictor import SAM2ImagePredictor
from llama_index.core.schema import ImageDocument, NodeRelationship

class ImageSegmentationModel(abc.ABC):
    @abc.abstractmethod
    def segment_image(self, image_node: ImageDocument, **kwargs) -> list[ImageRegion]:
        pass


class SamForImageSegmentation(ImageSegmentationModel):
    _default_configuration = {
        # "settings": {
        #     "points_per_side": 32,
        #     "points_per_batch": 128,
        #     "pred_iou_thresh": 0.7,
        #     "stability_score_thresh": 0.92,
        #     "stability_score_offset": 0.7,
        #     "crop_n_layers": 1,
        #     "box_nms_thresh": 0.7,
        #     "crop_n_points_downscale_factor": 2,
        #     "min_mask_region_area": 25.0,
        #     "use_m2m": True,
        # }
    },

    _model: Optional[SAM2ImagePredictor] = None
    
    def __init__(
            self,
            model_name: str,
            device: str = "cpu",
        ):
        self._model_name = model_name
        self._device = device


    def _get_or_create_sam2(self) -> SAM2ImagePredictor:
        """
        Retrieves or creates an instance of the Owlv2ForObjectDetection model.

        Returns:
            Owlv2ForObjectDetection: The object detection model instance.
        """
        if self._model is None:
            self._model = SAM2ImagePredictor.from_pretrained(self._model_name, device_map=self._device, device=self._device) #, **self._default_configuration)
        return self._model


    def segment_image(self, image_node: ImageDocument, bbox_list: list[ImageRegion] = None) -> list[ImageDocument]:
        """
        Parses an image node by cropping it into smaller image chunks based on the provided annotations.

        Args:
            image_node (Node): The image node to be parsed.
            configuration (dict): The configuration containing bounding box annotations.

        Returns:
            list[Node]: A list of image chunks generated from the cropping process.
        """
        img = Image.open(image_node.resolve_image()).convert("RGB")

        predictor = self._get_or_create_sam2()

        with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
            predictor.set_image(img)
                
            annotations = []
            for bbox in bbox_list:
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                annotations.append(predictor.predict(box=(x1, y1, x2, y2)))

            # Initialize a list to hold the generated image chunks
            image_chunks: list[Node] = []
            # Iterate over the annotations and corresponding bounding boxes
            for ann, bbox in zip(annotations, bbox_list):
                # Extract the coordinates of the bounding box
                box = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                # Find the index of the best mask in the annotation array
                best_mask_idx = ann[1].argmax().item()
                # Create a mask from the annotation array
                mask = Image.fromarray((ann[0][best_mask_idx] * 255).astype(np.uint8))
                # Composite the original image with a new RGB image using the mask
                masked_image = Image.composite(img, Image.new("RGB", img.size), mask)
                # Crop the masked image to the bounding box dimensions
                cropped_image = masked_image.crop(box)

                # Prepare metadata for the image chunk
                x1, y1, x2, y2 = box
                region = dict(x1=x1, y1=y1, x2=x2, y2=y2)
                metadata = dict(region=region)

                # Create an ImageNode from the cropped image and set its relationships
                image_chunk = ImageDocument(image=image_to_base64(cropped_image), image_mimetype=image_node.image_mimetype, metadata=metadata)
                image_chunk.relationships[NodeRelationship.SOURCE] = try_get_source_ref_node_info(image_node)
                image_chunk.relationships[NodeRelationship.PARENT] = image_node.as_related_node_info()
                # Append the created image chunk to the list
                image_chunks.append(image_chunk)
                # Send an event indicating that an image chunk has been generated

        children_collection = image_node.relationships.get(NodeRelationship.CHILD, [])
        image_node.relationships[NodeRelationship.CHILD] = children_collection + [c.as_related_node_info() for c in image_chunks[1:]]
        
        return image_chunks
