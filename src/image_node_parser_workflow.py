from io import BytesIO
from llama_index.core.schema import ImageDocument, ImageNode, NodeRelationship, RelatedNodeInfo, BaseNode, TextNode
from llama_index.core.workflow import Event,StartEvent,StopEvent,Workflow,step
from llama_index.core.workflow.errors import WorkflowRuntimeError
from llama_index.core.multi_modal_llms import MultiModalLLM
import logging
from PIL import Image
from sam2.automatic_mask_generator import SAM2ImagePredictor
from typing import Optional
import base64
import numpy as np
import shutil
import torch
import requests
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from .owl_v2 import Owlv2ProcessorWithNMS

class ImageRegion:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, label: str, score: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
        self.score = score

class ImageLoadedEvent(Event):
    image: ImageNode
    segmentation_configuration: dict | None
    object_detection_configuration: dict | None

class BBoxCreatedEvent(Event):
    """
    Event triggered when a bounding box is created for an image.

    Attributes:
        image (ImageNode): The image associated with the bounding box.
        segmentation_configuration (dict, optional): Configuration settings for image segmentation.
        object_detection_configuration (dict, optional): Configuration settings for object detection.
    """
    image: ImageNode
    segmentation_configuration: dict | None
    object_detection_configuration: dict | None

class ImageParsedEvent(Event):
    """
    Event triggered when an image has been successfully parsed.

    Attributes:
        source (ImageNode): The original image node that was parsed.
        chunks (list[ImageNode]): A list of image nodes representing the parsed chunks of the original image.
    """
    source: ImageNode
    chunks: list[ImageNode]

class ImageChunkGenerated(Event):
    """
    Event triggered when an image chunk is generated.

    Attributes:
        image_node (ImageNode): The image node representing the generated chunk.
    """
    image_node: ImageNode
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

    _default_predictor_configuration = {
        "model_name": "facebook/sam2-hiera-small",
        "sam_settings": {}
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
        #     "device_map": "cpu"
        # }
    },

    _object_detection_configuration = dict(
        confidence=0.1,
        nms_threshold=0.3
    )

    multi_modal_llm: Optional[MultiModalLLM] = None
    processor: Optional[AutoProcessor] = None
    model: Optional[Owlv2ForObjectDetection] = None

    def get_or_create_owl_v2(self) -> Owlv2ForObjectDetection:
        """
        Retrieves or creates an instance of the Owlv2ForObjectDetection model.

        Returns:
            Owlv2ForObjectDetection: The object detection model instance.
        """
        if self.model is None:
            self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")
        return self.model
    
    def get_or_create_owl_v2_processor(self) -> AutoProcessor:
        """
        Retrieves or creates an instance of the Owlv2ProcessorWithNMS processor.

        Returns:
            AutoProcessor: The processor instance for handling image processing.
        """
        if self.processor is None:
            self.processor = Owlv2ProcessorWithNMS.from_pretrained("google/owlv2-large-patch14-ensemble")
        return self.processor

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
        sam_configuration = self._default_predictor_configuration
        object_detection_configuration = self._object_detection_configuration
        if hasattr(start_event, "segmentation_configuration") and start_event.segmentation_configuration is not None:
            sam_configuration = start_event.segmentation_configuration
        if hasattr(start_event, "image") and start_event.image is not None and isinstance(start_event.image, ImageNode):
            return ImageLoadedEvent(image=start_event.image, segmentation_configuration=sam_configuration)
        elif hasattr(start_event, "base64_image") and start_event.base64_image is not None:
            document = ImageDocument(image=start_event.base64_image, mimetype=start_event.mimetype, image_mimetype=start_event.mimetype)
            return ImageLoadedEvent(image=document, segmentation_configuration=sam_configuration)
        elif hasattr(start_event, "image_path") and start_event.image_path is not None:
            image = Image.open(start_event.image_path).convert("RGB")
            document = ImageDocument(image=self.image_to_base64(image), mimetype="image/jpg", image_mimetype="image/jpg")
            return ImageLoadedEvent(image=document, segmentation_configuration=sam_configuration, object_detection_configuration=object_detection_configuration)
        else:
            return StopEvent()
        
    @step()
    async def create_bboxes(self, image_laoded_event: ImageLoadedEvent) -> BBoxCreatedEvent:
        """
        Create bounding boxes for the image based on the segmentation configuration.

        Args:
            image_laoded_event (ImageLoadedEvent): The event containing the loaded image and its segmentation configuration.

        Returns:
            BBoxCreatedEvent: The event containing the image and updated segmentation configuration.
            StopEvent: If bounding box creation fails due to an error.
        """
        try:
            if 'bbox_list' not in image_laoded_event.segmentation_configuration:
                if 'prompt' not in image_laoded_event.segmentation_configuration:
                    prompt = self.multi_modal_llm.complete(
                        "Find the most important entities in the image and produce a list of short prompts to use for an object detection model. Put each single prompt on a new line. Emit only the prompts.",
                        [image_laoded_event.image]
                    )
                    image_laoded_event.segmentation_configuration["prompt"] = prompt.text
                
                bbox_list = self._detect_bboxes_with_owlv2(
                    image_laoded_event.image,
                    image_laoded_event.segmentation_configuration['prompt'],
                    image_laoded_event.object_detection_configuration.get("confidence", 0.1),
                    image_laoded_event.object_detection_configuration.get("nms_threshold", 0.3)
                )
                
        except Exception as e:
            logging.error(f"Failed to create bounding boxes: {e}", exc_info=True)
            return StopEvent(reason="Bounding box creation failed due to an error.")

        return BBoxCreatedEvent(image=image_laoded_event.image, segmentation_configuration=image_laoded_event.segmentation_configuration)

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
        parsed: list[ImageNode] = []

        parsed = self._parse_image_node_with_sam2(bounding_boxes_created_event.image, bounding_boxes_created_event.segmentation_configuration)

        if len(parsed) == 0:
            result = {
                "source": bounding_boxes_created_event.image,
                "chunks": []
            }
            return StopEvent(result=result)
        else:
            return ImageParsedEvent(source=bounding_boxes_created_event.image, chunks=parsed)
        
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
                                mimetype=image_chunk.mimetype,
                                image_mimetype=image_chunk.mimetype
                            )
                        ],
                    )
                    # Create a TextNode to store the generated description
                    image_description_node = TextNode(
                        text=image_description.text,
                        mimetype="text/plain"
                    )
                    # Establish a relationship between the description node and the source image node
                    image_description_node.relationships[NodeRelationship.SOURCE] = self._ref_doc_id(image_parsed_event.source)
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

    def _detect_bboxes_with_owlv2(self, image_node: ImageNode, prompt: str, confidence: float, nms_threshold: float) -> list[ImageRegion]:
        """
        Detects bounding boxes in the image using the Owlv2 model.

        Args:
            image_node (ImageNode): The image node to process.
            prompt (str): The prompt for the object detection model.
            confidence (float): The confidence threshold for detections.
            nms_threshold (float): The non-maximum suppression threshold.

        Returns:
            list[ImageRegion]: A list of detected bounding boxes with their associated labels and scores.
        """
        image = Image.open(image_node.resolve_image()).convert("RGB")
        processor = self.get_or_create_owl_v2_processor()
        model = self.get_or_create_owl_v2()

        texts = [[x.strip() for x in prompt.split("\n")]]
        inputs = processor(text=texts, images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        def get_preprocessed_image(pixel_values):
            # Step 1: Remove the batch dimension from pixel_values and convert to numpy array
            pixel_values = pixel_values.squeeze().numpy()
            
            # Step 2: Unnormalize the image by applying the inverse of CLIP's normalization
            # Multiply by standard deviation and add mean
            unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
            
            # Step 3: Scale the pixel values to 0-255 range and convert to 8-bit unsigned integer
            unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
            
            # Step 4: Rearrange the color channel axis from first to last (CHW to HWC)
            unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
            
            # Step 5: Convert the numpy array to a PIL Image object
            unnormalized_image = Image.fromarray(unnormalized_image)
            return unnormalized_image

        unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        
        results = processor.post_process_object_detection_with_nms(
            outputs=outputs, threshold=confidence, nms_threshold=nms_threshold, target_sizes=target_sizes
        )

        i = 0
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        # Initialize an empty list to store ImageRegion objects
        annotations: list[ImageRegion] = [] 
        
        # Iterate through the detected boxes, scores, and labels
        for box, score, label in zip(boxes, scores, labels):
            # Round the box coordinates to 2 decimal places
            box = [round(i, 2) for i in box.tolist()]
            
            # Print detection information for debugging
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            
            # Unpack the box coordinates
            x1, y1, x2, y2 = box
            
            # Create an ImageRegion object with the detection information
            image_region = ImageRegion(x1, y1, x2, y2, label, score)
            
            # Add the ImageRegion to the annotations list
            annotations.append(image_region)
    
        return annotations

    def _parse_image_node_with_sam2(self, image_node: ImageNode, configuration: dict) -> list[ImageNode]:
        """
        Parses an image node by cropping it into smaller image chunks based on the provided annotations.

        Args:
            image_node (ImageNode): The image node to be parsed.
            configuration (dict): The configuration containing bounding box annotations.

        Returns:
            list[ImageNode]: A list of image chunks generated from the cropping process.
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
                    image_chunk = ImageNode(image=self.image_to_base64(cropped_image), mimetype=image_node.mimetype, metadata=metadata)
                    image_chunk.relationships[NodeRelationship.SOURCE] = self._ref_doc_id(image_node)
                    image_chunk.relationships[NodeRelationship.PARENT] = image_node.as_related_node_info()
                    # Append the created image chunk to the list
                    image_chunks.append(image_chunk)
                    # Send an event indicating that an image chunk has been generated
                    self.send_event(ImageChunkGenerated(image_node=image_chunk))
                except Exception as e:
                    # Handle any exceptions by sending a workflow runtime error event
                    self.send_event(WorkflowRuntimeError(e))
                    continue

        children_collection = image_node.relationships.get(NodeRelationship.CHILD, [])
        image_node.relationships[NodeRelationship.CHILD] = children_collection + [c.as_related_node_info() for c in image_chunks[1:]]
        
        return image_chunks

    def _ref_doc_id(self, node: BaseNode) -> RelatedNodeInfo:
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
        return image_str.decode('utf-8')  # Convert bytes to string