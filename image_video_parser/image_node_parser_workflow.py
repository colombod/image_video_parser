from llama_index.core.schema import ImageDocument, NodeRelationship, TextNode
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.core.workflow.errors import WorkflowRuntimeError
from llama_index.core.multi_modal_llms import MultiModalLLM
import logging
from PIL import Image
from sam2.automatic_mask_generator import SAM2ImagePredictor
from typing import Optional
import numpy as np
import torch
from PIL import Image
import numpy as np
import torch

from .object_segmentation_model import ImageSegmentationModel
from .object_detection_model import ObjectDetectionModel
from .utils import image_to_base64_string, try_get_source_ref_node_info, ImageRegion


class ImageLoadedEvent(Event):
    """
    Event triggered when an image is successfully loaded.

    This event carries information about the loaded image, including optional
    bounding box information and a prompt.

    Attributes:
        image (ImageDocument): The loaded image node.
        bbox_list (Optional[list[ImageRegion]]): A list of bounding boxes associated with the image, if any.
        prompt (Optional[str]): An optional prompt associated with the image.
    """
    image: ImageDocument
    bbox_list: Optional[list[ImageRegion]]
    prompt: Optional[str]


class BBoxCreatedEvent(Event):
    """
    Event triggered when a bounding box is created for an image.

    Attributes:
        image (ImageDocument): The image associated with the bounding box.
    """
    image: ImageDocument
    bbox_list: list[ImageRegion]


class ImageParsedEvent(Event):
    """
    Event triggered when an image has been successfully parsed.

    Attributes:
        source (ImageDocument): The original image node that was parsed.
        chunks (list[ImageDocument]): A list of image nodes representing the parsed chunks of the original image.
    """
    source: ImageDocument
    chunks: list[ImageDocument]


class ImageNodeParserWorkflow(Workflow):
    """
    Workflow for parsing images and generating bounding boxes and descriptions.

    This class handles the loading of images, creating bounding boxes using object detection,
    parsing the images into chunks, and generating descriptions for each chunk using a multi-modal
    language model.

    Attributes:
        _default_predictor_configuration (dict): Default configuration for the predictor model.
        _object_detection_configuration (dict): Configuration for object detection parameters.
        multi_modal_llm (Optional[MultiModalLLM]): The multi-modal language model used for generating prompts and descriptions.
        processor (Optional[AutoProcessor]): The processor for handling image processing tasks.
        model (Optional[Owlv2ForObjectDetection]): The object detection model.
    """

    multi_modal_llm: Optional[MultiModalLLM] = None
    object_detection_model: Optional[ObjectDetectionModel] = None
    image_segmentation_model: Optional[ImageSegmentationModel] = None

    @step()
    async def load_image(self, start_event: StartEvent) -> ImageLoadedEvent | StopEvent:
        """
        Load an image based on the provided event.

        Args:
            start_event (StartEvent): The event containing the image information.

        Returns:
            ImageLoadedEvent: The event containing the loaded image.
            StopEvent: If no valid image is provided.

        Raises:
            ValueError: If no image is provided.
        """

        bbox_list = start_event.get("bbox_list", None)
        prompt = start_event.get("prompt", None)

        image_document: ImageDocument = None

        # if hasattr(start_event, "image") and start_event.image is not None and isinstance(start_event.image, Node):
        #     image_document = start_event.image
        if hasattr(start_event, "image") and start_event.image is not None and isinstance(start_event.image, ImageDocument):
            image_document = start_event.image
        elif hasattr(start_event, "base64_image") and start_event.base64_image is not None:
            image_document = ImageDocument(image=start_event.base64_image, image_mimetype=start_event.mimetype)
        elif hasattr(start_event, "image_path") and start_event.image_path is not None:
            image = Image.open(start_event.image_path).convert("RGB")
            image_document = ImageDocument(image=image_to_base64_string(image), image_mimetype="image/jpg")
        else:
            return StopEvent()
        
        return ImageLoadedEvent(image=image_document, bbox_list=bbox_list, prompt=prompt)
        
    @step()
    async def create_bboxes(self, image_loaded_event: ImageLoadedEvent) -> BBoxCreatedEvent:
        """
        Create bounding boxes for the image based on the segmentation configuration.

        Args:
            image_laoded_event (ImageLoadedEvent): The event containing the loaded image and its segmentation configuration.

        Returns:
            BBoxCreatedEvent: The event containing the image and updated segmentation configuration.
            StopEvent: If bounding box creation fails due to an error.
        """
        try:
            bbox_list = image_loaded_event.bbox_list
            if bbox_list is None:
                prompt = image_loaded_event.prompt
                if prompt is None:
                    prompt = self.multi_modal_llm.complete(
                        "Find the most important entities (10 maximum) in the image and produce a list of short prompts to use for an object detection model. Give priorities to people, foreground elements, animals. Put each single prompt on a new line. Emit only the prompts without any punctuation.",
                        [image_loaded_event.image]
                    ).text
                
                bbox_list = self.object_detection_model.detect_bboxes(
                    image_loaded_event.image,
                    prompt=prompt,
                )

            return BBoxCreatedEvent(image=image_loaded_event.image, bbox_list=bbox_list)
                
        except Exception as e:
            logging.error(f"Failed to create bounding boxes: {e}", exc_info=True)
            return StopEvent(result="Bounding box creation failed due to an error.")


    @step()
    async def parse_image(self, bounding_boxes_created_event: BBoxCreatedEvent) -> ImageParsedEvent | StopEvent:
        """
        Parses the given image using the _parse_image_node_with_sam2 method.

        Args:
            bounding_boxes_created_event (BBoxCreatedEvent): The event containing the created bounding boxes and the image.

        Returns:
            ImageParsedEvent: The event containing the parsed image chunks.
            StopEvent: If no chunks are generated.
        """
        parsed: list[ImageDocument] = []
        image = bounding_boxes_created_event.image
        bbox_list = bounding_boxes_created_event.bbox_list

        parsed = self.image_segmentation_model.segment_image(image, bbox_list)

        if len(parsed) == 0:
            result = {
                "source": image,
                "chunks": []
            }
            return StopEvent(result=result)
        else:
            return ImageParsedEvent(source=image, chunks=parsed)
        
    @step()
    async def describe_image(self, image_parsed_event: ImageParsedEvent) -> StopEvent:
        """
        Generates descriptions for each chunk of the parsed image.

        This method iterates over the image chunks in the parsed event and uses a multi-modal
        language model to generate textual descriptions for each chunk. The descriptions are
        stored as TextNode instances with associated relationships to the source and parent nodes.

        Args:
            image_parsed_event (ImageParsedEvent): The event containing the parsed image and its chunks.

        Returns:
            StopEvent: An event containing the source image, the image chunks, and their corresponding descriptions.
        """
        image_descriptions: list[TextNode] = []
        
        # Check if a multi-modal language model is available
        if self.multi_modal_llm is not None:
            # Iterate over each chunk of the parsed image
            for image_chunk in image_parsed_event.chunks:
                try:
                    # Use the multi-modal language model to generate a description for the image chunk
                    image_description = self.multi_modal_llm.complete(
                        prompt="Describe the image above in a few words.",
                        image_documents=[
                            ImageDocument(
                                image=image_chunk.image,
                                image_mimetype=image_chunk.image_mimetype
                            )
                        ],
                    )
                    # Create a TextNode to store the generated description
                    image_description_node = TextNode(
                        text=image_description.text,
                        mimetype="text/plain"
                    )
                    # Establish a relationship between the description node and the source image node
                    image_description_node.relationships[NodeRelationship.SOURCE] = try_get_source_ref_node_info(image_parsed_event.source)
                    # Establish a parent relationship to the current image chunk
                    image_description_node.relationships[NodeRelationship.PARENT] = image_chunk.as_related_node_info()
                    # Append the description node to the list of image descriptions
                    image_descriptions.append(image_description_node)
                except Exception:
                    # If an error occurs during description generation, append None to maintain list integrity
                    image_descriptions.append(None)
          
        result = {
            "source": image_parsed_event.source,
            "chunks": image_parsed_event.chunks,
            "descriptions": image_descriptions
        }

        return StopEvent(result=result)

    def _parse_image_node_with_sam2(self, image_node: ImageDocument, configuration: dict) -> list[ImageDocument]:
        """
        Parses an image node by cropping it into smaller image chunks based on the provided annotations.

        Args:
            image_node (Node): The image node to be parsed.
            configuration (dict): The configuration containing bounding box annotations.

        Returns:
            list[Node]: A list of image chunks generated from the cropping process.
        """
        img = Image.open(image_node.resolve_image()).convert("RGB")

        sam_settings = configuration.get("sam_settings", {})

        predictor = SAM2ImagePredictor.from_pretrained(configuration["model_name"], device="cpu", **sam_settings)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(img)
                
            annotations = []
            for bbox in configuration["bbox_list"]:
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                annotations.append(predictor.predict(box=(x1, y1, x2, y2)))

            # Initialize a list to hold the generated image chunks
            image_chunks = []
            # Iterate over the annotations and corresponding bounding boxes
            for ann, bbox in zip(annotations, configuration["bbox_list"]):
                # Extract the coordinates of the bounding box
                box = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                # Create a mask from the annotation array
                mask = Image.fromarray(ann[0][-1].astype(np.uint8))
                # Composite the original image with a new RGB image using the mask
                masked_image = Image.composite(img, Image.new("RGB", img.size), mask)
                # Crop the masked image to the bounding box dimensions
                cropped_image = masked_image.crop(box)

                # Prepare metadata for the image chunk
                x1, y1, x2, y2 = box
                region = dict(x1=x1, y1=y1, x2=x2, y2=y2)
                metadata = dict(region=region)
                try:
                    # Create an ImageNode from the cropped image and set its relationships
                    image_chunk = ImageDocument(image=image_to_base64_string(cropped_image), image_mimetype=image_node.image_mimetype, metadata=metadata)
                    image_chunk.relationships[NodeRelationship.SOURCE] = try_get_source_ref_node_info(image_node)
                    image_chunk.relationships[NodeRelationship.PARENT] = image_node.as_related_node_info()
                    # Append the created image chunk to the list
                    image_chunks.append(image_chunk)
                except Exception as e:
                    # Handle any exceptions by sending a workflow runtime error event
                    self.send_event(WorkflowRuntimeError(e))
                    continue

        children_collection = image_node.relationships.get(NodeRelationship.CHILD, [])
        image_node.relationships[NodeRelationship.CHILD] = children_collection + [c.as_related_node_info() for c in image_chunks[1:]]
        
        return image_chunks