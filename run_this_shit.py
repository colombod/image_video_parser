from typing import Optional
from llama_index.core.schema import ImageDocument, ImageNode, NodeRelationship, RelatedNodeInfo
from PIL import Image
from io import BytesIO
import base64
import torch
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

predictor = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-small", device_map="cpu")

image = Image.open("./maestro_domenico_bini_famoso_sul_web_e_nel_mondo.jpeg")

def image_to_base64(pil_image, format="JPEG"):
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    image_str = base64.b64encode(buffered.getvalue())
    return image_str.decode('utf-8') # Convert bytes to string

mime_type = "image/jpg"

document = ImageDocument(image=image_to_base64(image), mime_type=mime_type, image_mime_type=mime_type)
top_level_node = ImageNode(image=document.image, mime_type=document.mime_type)
top_level_node.relationships[NodeRelationship.SOURCE] = document.as_related_node_info()

def parse_image_node(image_node: ImageNode) -> list[ImageNode]:
    img = image_node.image

    with torch.inference_mode(): #, torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(img)
        annotations = predictor.generate(img) # do this if we don't already have a grid

        # from each mask crop the image
        cropped_images = []
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
            left, top, right, bottom = ann["crop_box"]
            # cropped_image = img.crop((left, top, right, bottom))
            cropped_image = image[top:bottom, left:right].copy()
            cropped_images.append((cropped_image,  dict(x=left, y=top, height=bottom-top, width=right-left)))

        image_chunks = [image_node]
        for c, region in cropped_images:
            metadata = dict(region=region)
            image_chunk = ImageNode(image=image_to_base64(c), mime_type=image_node.mime_type, metadata=metadata)
            image_chunk.relationships[NodeRelationship.SOURCE] = ref_doc_id(image_node)
            image_chunk.relationships[NodeRelationship.PARENT] = image_node.as_related_node_info()
            image_chunks.append(image_chunk)

        children_collection = image_node.relationships.get(NodeRelationship.CHILDREN, [])
        image_node.relationships[NodeRelationship.CHILDREN] = children_collection + [c.as_related_node_info() for c in image_chunks[1:]]
        return image_chunks

def ref_doc_id(node: ImageNode) -> RelatedNodeInfo:
    """Deprecated: Get ref doc id."""
    source_node = node.source_node
    if source_node is None:
        return node.as_related_node_info()
    return source_node


parsed_nodes = parse_image_node(top_level_node)
