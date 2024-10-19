import base64
from PIL import Image
from io import BytesIO
from llama_index.core.schema import RelatedNodeInfo, BaseNode


class ImageRegion:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, label: str, score: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
        self.score = score


def ref_doc_id(node: BaseNode) -> RelatedNodeInfo:
    """
    Returns the related node information of the document for the given ImageNode.

    Args:
        node (ImageNode): The ImageNode for which to retrieve the related node information.

    Returns:
        RelatedNodeInfo: The related node information for the given ImageNode.
    """
    source_node = node.source_node
    if source_node is None:
        return node.as_related_node_info()
    return source_node

def image_to_base64(pil_image: Image, format="JPEG"):
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
    return image_str.decode('utf-8')  # Convert bytes to string