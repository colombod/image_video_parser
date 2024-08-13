from typing import Optional
from llamaindex.core.schema import ImageDocument, ImageNode, NodeRelationship, RelatedNodeInfo
from PIL import Image
from io import BytesIO
import base64
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

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

def do_shit(image_node: ImageNode) -> list[ImageNode]:
    img = image_node.image

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(img)
        masks, _, _ = predictor.predict(img)

        # from each mask crop the image
        cropped_images = []
        for mask in masks:
            cropped_image = img.copy()
            cropped_image = cropped_image.crop(mask)
            cropped_images.append(cropped_image)

        image_chunks = [image_node]
        for c in cropped_images:
            region = dict(x=, y=, height=, width=)
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


parsed_nodes = do_shit(top_level_node)
