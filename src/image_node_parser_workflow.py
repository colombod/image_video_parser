from io import BytesIO
from llama_index.core.schema import ImageDocument, ImageNode, NodeRelationship, RelatedNodeInfo, BaseNode, TextNode
from llama_index.core.workflow import Event,StartEvent,StopEvent,Workflow,step
from llama_index.core.workflow.errors import WorkflowRuntimeError
from llama_index.core.multi_modal_llms import MultiModalLLM
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
    def __init__(self, x: int, y: int, width: int, height: int, label: str, score: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.score = score

class ImageLoadedEvent(Event):
    image: ImageNode
    segmentation_configuration: dict | None
    object_detection_configuration: dict | None

class ImageParsedEvent(Event):
    source: ImageNode
    chunks: list[ImageNode]

class ImageChunkGenerated(Event):
    image_node: ImageNode

class ImageNodeParserWorkflow(Workflow):
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
        if self.model is None:
            self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")
        return self.model
    
    def get_or_create_owl_v2_processor(self) -> AutoProcessor:
        if self.processor is None:
            self.processor = Owlv2ProcessorWithNMS.from_pretrained("google/owlv2-large-patch14-ensemble")
        return self.processor

    @step()
    async def load_image(self, ev: StartEvent) -> ImageLoadedEvent|StopEvent:
        """
        Load an image based on the provided event.

        Args:
            ev (StartEvent): The event containing the image information.

        Returns:
            ImageLaodedEvent: The event containing the loaded image.

        Raises:
            ValueError: If no image is provided.
        """

        sam_configuration = self._default_predictor_configuration
        object_detection_configuration = self._object_detection_configuration
        if hasattr(ev, "segmentation_configuration") and ev.segmentation_configuration is not None:
            sam_configuration = ev.segmentation_configuration
        if hasattr(ev, "image") and ev.image is not None and isinstance(ev.image, ImageNode):
            return ImageLoadedEvent(image=ev.image, segmentation_configuration=sam_configuration)
        elif hasattr(ev, "base64_image") and ev.base64_image is not None:
            document = ImageDocument(image=ev.base64_image, mimetype=ev.mimetype, image_mimetype=ev.mimetype)
            return ImageLoadedEvent(image=document, segmentation_configuration=sam_configuration)
        elif hasattr(ev, "image_path") and ev.image_path is not None:
            image = Image.open(ev.image_path).convert("RGB")
            document = ImageDocument(image=self.image_to_base64(image), mimetype="image/jpg", image_mimetype="image/jpg")
            return ImageLoadedEvent(image=document, segmentation_configuration=sam_configuration, object_detection_configuration=object_detection_configuration)
        else:
            return StopEvent()

    @step()
    async def parse_image(self, ev: ImageLoadedEvent) -> ImageParsedEvent | StopEvent:
        """
        Parses the given image using the _parse_image_node_with_sam2 method.
        Parameters:
            ev (ImageLaodedEvent): The event containing the loaded image.
        Returns:
            StopEvent: The event containing the parsed image chunks.
        """
        parsed: list[ImageNode] = []

        if 'bbox_list' not in ev.segmentation_configuration:
            if 'prompt' not in ev.segmentation_configuration:
                prompt = self.multi_modal_llm.complete("Find the most important entities in the image and produce a list of short prompts to use for an object detection model. Put each single prompt on a new line. Emit only the prompts.", [ev.image])
                ev.segmentation_configuration["prompt"] = prompt.text
            
            bbox_list = self._detect_bboxes_with_owlv2(
                ev.image,
                ev.segmentation_configuration['prompt'],
                ev.object_detection_configuration.get("confidence", 0.1),
                ev.object_detection_configuration.get("nms_threshold", 0.3)
            )
            ev.segmentation_configuration["bbox_list"] = bbox_list

        parsed = self._parse_image_node_with_sam2(ev.image, ev.segmentation_configuration)

        if len(parsed) == 0:
            result = {
                "source": ev.image,
                "chunks": []
            }
            return StopEvent(result=result)
        else:
            return ImageParsedEvent(source=ev.image, chunks=parsed)
        
    @step()
    async def describe_image(self, ev: ImageParsedEvent) ->  StopEvent:

        image_descriptions : list[TextNode]= []        
       
        if self.multi_modal_llm is not None:
            for image_chunk in ev.chunks:
                try:
                    image_description =  self.multi_modal_llm.complete(
                        prompt="Describe the image above in a few words.",
                        image_documents=[ImageDocument(image=image_chunk.image, mimetype=image_chunk.mimetype, image_mimetype=image_chunk.mimetype)],
                    )
                    image_description_node = TextNode(text=image_description.text, mimetype="text/plain")
                    image_description_node.relationships[NodeRelationship.SOURCE] = self._ref_doc_id(ev.source)
                    image_description_node.relationships[NodeRelationship.PARENT] = image_chunk.as_related_node_info()
                    image_descriptions.append(image_description_node)
                except Exception:
                    image_descriptions.append(None)
              
        result = {
            "source": ev.source,
            "chunks": ev.chunks,
            "descriptions": image_descriptions
        }

        return StopEvent(result=result)


    def _detect_bboxes_with_owlv2(self, image_node: ImageNode, prompt: str, confidence: float, nms_threshold: float) -> list[ImageRegion]:
        """
        Detects stuff and returns the annotated image.
        Parameters:
            image: The input image (as numpy array).
            seg_input: The segmentation input (i.e. the prompt for the model).
            debug (bool): Flag to enable logging for debugging purposes.
        Returns:
            tuple: (numpy array of image, list of (label, (x1, y1, x2, y2)) tuples)
        """
    
        # Step 2: Detect stuff using owl_v2

        image = Image.open(image_node.resolve_image()).convert("RGB")
        processor = self.get_or_create_owl_v2_processor()
        model = self.get_or_create_owl_v2()


        texts = [[x.strip() for x in prompt.split("\n")]]
        inputs = processor(text=texts, images=image, return_tensors="pt")

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Note: boxes need to be visualized on the padded, unnormalized image
        # hence we'll set the target image sizes (height, width) based on that
        def get_preprocessed_image(pixel_values):
            pixel_values = pixel_values.squeeze().numpy()
            unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
            unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
            unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
            unnormalized_image = Image.fromarray(unnormalized_image)
            return unnormalized_image

        unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        results = processor.post_process_object_detection_with_nms(
            outputs=outputs, threshold=confidence, nms_threshold=nms_threshold, target_sizes=target_sizes
        )

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        # Prepare annotations for AnnotatedImage output
        annotations: list[ImageRegion] = [] 
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            x1, y1, x2, y2 = box
            image_region = ImageRegion(x1, y1, x2-x1, y2-y1, label, score)
            annotations.append(image_region)
    
        return annotations


    def _parse_image_node_with_sam2(self, image_node: ImageNode, configuration : dict) -> list[ImageNode]:
        """
        Parses an image node by cropping it into smaller image chunks based on the provided annotations.
        Args:
            image_node (ImageNode): The image node to be parsed.
        Returns:
            list[ImageNode]: A list of image chunks generated from the cropping process.
        """
        img = np.array(Image.open(image_node.resolve_image()).convert("RGB"))

        sam_settings = configuration.get("sam_settings", {})

        predictor = SAM2ImagePredictor.from_pretrained(configuration["model_name"], device="cpu", **sam_settings)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(img)
                
            annotations = []
            for bbox in configuration["bbox_list"]:
                x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
                right = x + width
                bottom = y + height
                annotations.append(predictor.predict(box=(x, y, right, bottom)))

            # from each mask crop the image
            image_chunks = []
            for ann, bbox in zip(annotations, configuration["bbox_list"]):
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
                # left, top, width, height = ann["bbox"]
                # cropped_image = img.crop((left, top, right, bottom))
                x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
                right = x + width
                bottom = y + height
                img_clone = img.copy()
                img_clone = img_clone*ann[0][-1][..., None] # we might want to add back the alpha channel here
                cropped_image = img_clone[int(y):int(y+height), int(x):int(x+width)].copy()
                
                region = dict(x=x, y=y, height=height, width=width)
                metadata = dict(region=region)
                try:
                    image_chunk = ImageNode(image=self.image_to_base64(Image.fromarray(cropped_image.astype(np.uint8))), mimetype=image_node.mimetype, metadata=metadata)
                    image_chunk.relationships[NodeRelationship.SOURCE] = self._ref_doc_id(image_node)
                    image_chunk.relationships[NodeRelationship.PARENT] = image_node.as_related_node_info()
                    image_chunks.append(image_chunk)
                    self.send_event(ImageChunkGenerated(image_node=image_chunk))
                except Exception as e:
                    self.send_event(WorkflowRuntimeError(e))
                    continue

        children_collection = image_node.relationships.get(NodeRelationship.CHILD, [])
        image_node.relationships[NodeRelationship.CHILD] = children_collection + [c.as_related_node_info() for c in image_chunks[1:]]
        
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