from typing import Optional
from llama_index.core.schema import ImageDocument, ImageNode, NodeRelationship, RelatedNodeInfo
from PIL import Image
from io import BytesIO
import shutil
import base64
import torch
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

predictor = SAM2AutomaticMaskGenerator.from_pretrained(
    "facebook/sam2-hiera-small",
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
    device_map="cpu"
)

image = Image.open("./images/il_vulcano_3.png").convert("RGB")

def image_to_base64(pil_image, format="JPEG"):
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    image_str = base64.b64encode(buffered.getvalue())
    return image_str.decode('utf-8') # Convert bytes to string

mimetype = "image/jpg"

document = ImageDocument(image=image_to_base64(image), mimetype=mimetype, image_mimetype=mimetype)
top_level_node = ImageNode(image=document.image, mimetype=document.mimetype)
top_level_node.relationships[NodeRelationship.SOURCE] = document.as_related_node_info()

def parse_image_node(image_node: ImageNode) -> list[ImageNode]:
    img = np.array(Image.open(image_node.resolve_image()).convert("RGB"))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        annotations = predictor.generate(img) # do this if we don't already have a grid

        # from each mask crop the image
        image_chunks = [image_node]
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
                image_chunk = ImageNode(image=image_to_base64(Image.fromarray(cropped_image.astype(np.uint8))), mimetype=image_node.mimetype, metadata=metadata)
                image_chunk.relationships[NodeRelationship.SOURCE] = ref_doc_id(image_node)
                image_chunk.relationships[NodeRelationship.PARENT] = image_node.as_related_node_info()
                image_chunks.append(image_chunk)
            except Exception as e:
                print(e)
                continue


        children_collection = image_node.relationships.get(NodeRelationship.CHILD, [])
        image_node.relationships[NodeRelationship.CHILD] = children_collection + [c.as_related_node_info() for c in image_chunks[1:]]
        return image_chunks

def ref_doc_id(node: ImageNode) -> RelatedNodeInfo:
    """Deprecated: Get ref doc id."""
    source_node = node.source_node
    if source_node is None:
        return node.as_related_node_info()
    return source_node


parsed_nodes = parse_image_node(top_level_node)
# remove the ./output folder
shutil.rmtree("./output", ignore_errors=True)
shutil.os.mkdir("./output")
for node in parsed_nodes:
    # save node to folder ./output
    Image.open(node.resolve_image()).save(f"./output/{node.node_id}.png")
