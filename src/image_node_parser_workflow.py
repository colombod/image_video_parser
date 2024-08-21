from io import BytesIO
from llama_index.core.schema import ImageDocument, ImageNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.workflow import Event,StartEvent,StopEvent,Workflow,step
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from typing import Optional
import base64
import numpy as np
import shutil
import torch

class ImageLaodedEvent(Event):
    image: ImageNode

class ImageParsedEvent(Event):
    source: ImageNode
    chunks: list[ImageNode]

class ImageChunkGenerated:
    imageNode: ImageNode

class ImageNodeParserWorklof(Workflow):
    _predictor: SAM2AutomaticMaskGenerator

    def _init_(self,predictor: SAM2AutomaticMaskGenerator):
        self._predictor = predictor

    @step()
    async def load_image(self, ev: StartEvent) -> ImageLaodedEvent:
        if ev.image is not None and isinstance(ev.image, ImageNode):
            return ImageLaodedEvent(image=ev.image)
        elif ev.base64_image is not None:
            image = ImageDocument(image=ev.base64_image, mimetype=ev.mimetype, image_mimetype=ev.mimetype)
            top_level_node = ImageNode(image=image.image, mimetype=image.mimetype)
            top_level_node.relationships[NodeRelationship.SOURCE] = image.as_related_node_info()
            return ImageLaodedEvent(image=top_level_node)
        else:
            raise ValueError("No image provided")

    @step()
    async def parse_image(self, ev: ImageLaodedEvent) -> StopEvent:
        parsed = self._parse_image_node(ev.image)

        return StopEvent(image=ev.image, chunks=parsed)

    def _parse_image_node(self, image_node: ImageNode) -> list[ImageNode]:
        img = np.array(Image.open(image_node.resolve_image()).convert("RGB"))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            annotations = self._predictor.generate(img) # do this if we don't already have a grid

            # from each mask crop the image
            image_chunks = []
            for ann in annotations:
                # ann = {
                #     "segmentation": mask_data["segmentations"][idx],
                #     "area": area_from_rle(mask_data["rles"][idx]),
                #     "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                #     "predicted_iou": mask_data["iou_preds"][idx].item(),
                #     "point_coords": [mask_data["points"][idx].tolist()],
                #     "stability_score": mask_data["stability_score"][idx].item(),
                #     "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
                # }

                # crop the image with the annotation provided
                left, top, width, height = ann["bbox"]
                # cropped_image = img.crop((left, top, right, bottom))
                img_clone = img.copy()
                img_clone = img_clone*ann["segmentation"][..., None] # we might want to add back the alpha channel here
                cropped_image = img_clone[int(top):int(top+height), int(left):int(left+width)].copy()
                
                region = dict(x=left, y=top, height=height, width=width)
                metadata = dict(region=region)
                try:
                    image_chunk = ImageNode(image=self.image_to_base64(Image.fromarray(cropped_image.astype(np.uint8))), mimetype=image_node.mimetype, metadata=metadata)
                    image_chunk.relationships[NodeRelationship.SOURCE] = self._ref_doc_id(image_node)
                    image_chunk.relationships[NodeRelationship.PARENT] = image_node.as_related_node_info()
                    image_chunks.append(image_chunk)
                    self.send_event(ImageChunkGenerated(imageNode=image_chunk))
                except Exception as e:
                    print(e)
                    continue


        children_collection = image_node.relationships.get(NodeRelationship.CHILD, [])
        image_node.relationships[NodeRelationship.CHILD] = children_collection + [c.as_related_node_info() for c in image_chunks[1:]]
        self.send_event(ImageParsedEvent(source=image_node, chunks=image_chunks))
        return image_chunks

    def _ref_doc_id(self, node: ImageNode) -> RelatedNodeInfo:
        """Deprecated: Get ref doc id."""
        source_node = node.source_node
        if source_node is None:
            return node.as_related_node_info()
        return source_node
    def image_to_base64(self, pil_image, format="JPEG"):
        buffered = BytesIO()
        pil_image.save(buffered, format=format)
        image_str = base64.b64encode(buffered.getvalue())
        return image_str.decode('utf-8') # Convert bytes to string