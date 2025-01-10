import base64
from PIL import Image
from io import BytesIO
from llama_index.core.schema import RelatedNodeInfo, BaseNode, MediaResource, Node
import requests

class ImageRegion:
    """
    Represents a region within an image, typically used for object detection or segmentation tasks.

    This class encapsulates information about a rectangular region in an image, including its
    coordinates, label, and confidence score.

    Attributes:
        x1 (int): The x-coordinate of the top-left corner of the region.
        y1 (int): The y-coordinate of the top-left corner of the region.
        x2 (int): The x-coordinate of the bottom-right corner of the region.
        y2 (int): The y-coordinate of the bottom-right corner of the region.
        label (str): A descriptive label for the detected object or region.
        score (float): A confidence score associated with the detection, typically between 0 and 1.
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int, label: str, score: float):
        """
        Initializes an ImageRegion instance.

        Args:
            x1 (int): The x-coordinate of the top-left corner.
            y1 (int): The y-coordinate of the top-left corner.
            x2 (int): The x-coordinate of the bottom-right corner.
            y2 (int): The y-coordinate of the bottom-right corner.
            label (str): The label of the detected object or region.
            score (float): The confidence score of the detection.
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
        self.score = score

def try_get_source_ref_node_info(node: BaseNode) -> RelatedNodeInfo:
    """
    Retrieves the RelatedNodeInfo for the source of the given node.

    This function attempts to get the source node of the input node. If the source node
    exists, it returns its RelatedNodeInfo. Otherwise, it returns the RelatedNodeInfo
    of the input node itself.

    Args:
        node (BaseNode): The node for which to retrieve the source reference information.

    Returns:
        RelatedNodeInfo: The RelatedNodeInfo of the source node if it exists,
                         otherwise the RelatedNodeInfo of the input node.
    """
 

    source_node = node.source_node
    if source_node is None:
        return node.as_related_node_info()
    return source_node.as_related_node_info()

def image_to_raw_bytes(pil_image: Image, format="JPEG") -> bytes:
    """
    Converts a PIL image to raw bytes.

    Args:
        pil_image (PIL.Image.Image): The PIL image object to be converted.
        format (str, optional): The format of the image. Defaults to "JPEG".

    Returns:
        bytes: The raw bytes of the image.
    """
    # Create a BytesIO object to store the image data in memory
    buffered = BytesIO()
    
    # Save the PIL image to the BytesIO object in the specified format
    pil_image.save(buffered, format=format)
    
    # Get the raw bytes from the BytesIO object
    return buffered.getvalue()

def image_to_base64_binary(pil_image: Image, format="JPEG") -> bytes:
    """
    Converts a PIL image to base64 binary.

    Args:
        pil_image (PIL.Image.Image): The PIL image object to be converted.
        format (str, optional): The format of the image. Defaults to "JPEG".

    Returns:
        BytesIO: A BytesIO object containing the base64 encoded image data.
    """
    bytes = image_to_raw_bytes(pil_image, format=format)
    
    # Return the BytesIO object
    return base64.b64encode(bytes)

def image_to_base64_string(pil_image: Image, format="JPEG") -> str:
    """
    Converts a PIL image to base64 string.

    Args:
        pil_image (PIL.Image.Image): The PIL image object to be converted.
        format (str, optional): The format of the image. Defaults to "JPEG".

    Returns:
        str: The base64 encoded string representation of the image.
    """

    base64_raw = image_to_base64_binary(pil_image, format=format)
    
    # Decode the base64 bytes to a UTF-8 string and return
    return base64_raw.decode('utf-8')

def resolve_image(node: Node) -> BytesIO:
    """
    Resolves the image data from the node.

    This function resolves the image data from the node, which can be in the form of a URL,
    a base64 encoded string, or raw binary data.

    Args:
        node (BaseNode): The node containing the image data.

    Returns:
        BytesIO: A BytesIO object containing the image data.
    """
    media_resource = node.image_resource
    # Check if the media resource is available
    if media_resource is None:
        raise ValueError("Node does not contain image data.")
    return resolve_image(media_resource = media_resource)

def resolve_image(media_resource: MediaResource) -> bytes:
    """
    Resolves the image from the media resource.

    Args:
        media_resource (MediaResource): The media resource containing the image data.

    Returns:
        Image: base64 binary representation of the Image
    """
    if media_resource.data:
        return BytesIO(media_resource.data).getvalue()
    if media_resource.url:
        response = requests.get(media_resource.url)
        return BytesIO(response.content).getvalue()
    if media_resource.path:
        return BytesIO(base64.b64decode(open(media_resource.path, "rb"))).getvalue()
    
def create_node_from_image(image: Image) -> Node:
    """
    Creates a new node from an image and an existing node.

    Args:
        image (Image): The image to be associated with the new node.
        node (Node): The existing node to be used as a template for the new node.

    Returns:
        Node: The new node created from the image and the existing node.
    """
    new_node = Node(
        image_resource=MediaResource(data=image_to_base64_binary(image))
    )
    return new_node