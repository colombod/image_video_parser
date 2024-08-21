from io import BytesIO
from llama_index.core.schema import ImageDocument, ImageNode, NodeRelationship, RelatedNodeInfo, BaseNode
from llama_index.core.workflow import Event,StartEvent,StopEvent,Workflow,step
from llama_index.core.workflow.errors import WorkflowRuntimeError
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
    async def load_image(self, ev: StartEvent) -> ImageLaodedEvent|StopEvent:
        """
        Load an image based on the provided event.

        Args:
            ev (StartEvent): The event containing the image information.

        Returns:
            ImageLaodedEvent: The event containing the loaded image.

        Raises:
            ValueError: If no image is provided.
        """
        if ev.image is not None and isinstance(ev.image, ImageNode):
            return ImageLaodedEvent(image=ev.image)
        elif ev.base64_image is not None:
            image = ImageDocument(image=ev.base64_image, mimetype=ev.mimetype, image_mimetype=ev.mimetype)
            return ImageLaodedEvent(image=image)
        elif ev.image_path is not None:
            image = Image.open("./images/il_vulcano_3.png").convert("RGB")
            document = ImageDocument(image=self.image_to_base64(image), mimetype="image/jpg", image_mimetype="image/jpg")
            return ImageLaodedEvent(image=image)
        else:
            return StopEvent()

    @step()
    async def parse_image(self, ev: ImageLaodedEvent) -> StopEvent:
        """
        Parses the given image using the _parse_image_node method.
        Parameters:
            ev (ImageLaodedEvent): The event containing the loaded image.
        Returns:
            StopEvent: The event containing the parsed image chunks.
        """
        parsed = self._parse_image_node(ev.image)

        return StopEvent(image=ev.image, chunks=parsed)

    def _parse_image_node(self, image_node: ImageNode) -> list[ImageNode]:
        """
        Parses an image node by cropping it into smaller image chunks based on the provided annotations.
        Args:
            image_node (ImageNode): The image node to be parsed.
        Returns:
            list[ImageNode]: A list of image chunks generated from the cropping process.
        """
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
                    self.send_event(WorkflowRuntimeError(e))
                    continue

        children_collection = image_node.relationships.get(NodeRelationship.CHILD, [])
        image_node.relationships[NodeRelationship.CHILD] = children_collection + [c.as_related_node_info() for c in image_chunks[1:]]
        self.send_event(ImageParsedEvent(source=image_node, chunks=image_chunks))
        return image_chunks

    def _ref_doc_id(self, node: BaseNode) -> RelatedNodeInfo:
        """
        Returns the related node information of the document for the given ImageNode.

        Parameters:
            node (ImageNode): The ImageNode for which to retrieve the related node information.

        Returns:
            RelatedNodeInfo: The related node information for the given ImageNode.
        """
        source_node = node.source_node
        if source_node is None:
            return node.as_related_node_info()
        return source_node
    
    def image_to_base64(self, pil_image, format="JPEG"):
        """
        Converts a PIL image to base64 string.

        Args:
            pil_image (PIL.Image.Image): The PIL image object to be converted.
            format (str, optional): The format of the image. Defaults to "JPEG".

        Returns:
            str: The base64 encoded string representation of the image.
        """
        buffered = BytesIO()
        pil_image.save(buffered, format=format)
        image_str = base64.b64encode(buffered.getvalue())
        return image_str.decode('utf-8') # Convert bytes to string