import abc
from typing import Literal, Optional
import torch
import numpy as np
from PIL import Image
from llama_index.core.schema import ImageDocument
from transformers import AutoProcessor, Owlv2ForObjectDetection, AutoProcessor, AutoModelForCausalLM
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from .utils import ImageRegion
from .owl_v2 import Owlv2ProcessorWithNMS

class ObjectDetectionModel(abc.ABC):
    @abc.abstractmethod
    def detect_bboxes(self, image_node: ImageDocument, **kwargs) -> list[ImageRegion]:
        pass


class OwlV2ObjectDetectionModel(ObjectDetectionModel):
    _processor: Optional[AutoProcessor] = None
    _model: Optional[Owlv2ForObjectDetection] = None
    
    def __init__(
            self,
            confidence=0.1,
            nms_threshold=0.3,
            save_cropped_images: bool = False,
            output_dir: str = "./output",
            device: str = "cpu",
        ):
        self._owl_v2 = None
        self._owl_v2_processor = None
        self._save_cropped_images = save_cropped_images
        self._output_dir = output_dir
        self._confidence = confidence
        self._nms_threshold = nms_threshold
        self._device = device


    def _get_or_create_owl_v2(self) -> Owlv2ForObjectDetection:
        """
        Retrieves or creates an instance of the Owlv2ForObjectDetection model.

        Returns:
            Owlv2ForObjectDetection: The object detection model instance.
        """
        if self._model is None:
            self._model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", device_map=self._device).eval()
        return self._model
    
    def _get_or_create_owl_v2_processor(self) -> AutoProcessor:
        """
        Retrieves or creates an instance of the Owlv2ProcessorWithNMS processor.

        Returns:
            AutoProcessor: The processor instance for handling image processing.
        """
        if self._processor is None:
            self._processor = Owlv2ProcessorWithNMS.from_pretrained("google/owlv2-base-patch16-ensemble")
        return self._processor

    def detect_bboxes(self, image_node: ImageDocument, prompt: str, score_threshold: float = 0.1, **kwargs) -> list[ImageRegion]:
        """
        Detects bounding boxes in the image using the Owlv2 model.

        Args:
            image_node (ImageDocument): The image node to process.
            prompt (str): The prompt for the object detection model.
            confidence (float): The confidence threshold for detections.
            nms_threshold (float): The non-maximum suppression threshold.

        Returns:
            list[ImageRegion]: A list of detected bounding boxes with their associated labels and scores.
        """
        image = Image.open(image_node.resolve_image()).convert("RGB")
        processor = self._get_or_create_owl_v2_processor()
        model = self._get_or_create_owl_v2()

        texts = [[x.strip() for x in prompt.split("\n")]]
        inputs = processor(text=texts, images=image, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = model(**inputs)

        # def get_preprocessed_image(pixel_values):
        #     # Step 1: Remove the batch dimension from pixel_values and convert to numpy array
        #     pixel_values = pixel_values.squeeze().numpy()
            
        #     # Step 2: Unnormalize the image by applying the inverse of CLIP's normalization
        #     # Multiply by standard deviation and add mean
        #     unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
            
        #     # Step 3: Scale the pixel values to 0-255 range and convert to 8-bit unsigned integer
        #     unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
            
        #     # Step 4: Rearrange the color channel axis from first to last (CHW to HWC)
        #     unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
            
        #     # Step 5: Convert the numpy array to a PIL Image object
        #     unnormalized_image = Image.fromarray(unnormalized_image)
        #     return unnormalized_image

        # unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        # target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        
        # results = processor.post_process_object_detection_with_nms(
        #     outputs=outputs, threshold=self._confidence, nms_threshold=self._nms_threshold, target_sizes=target_sizes
        # )

        # i = 0
        # text = texts[i]
        # boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        size = max(image.size[:2])
        target_sizes = torch.Tensor([[size, size]])

        outputs.logits = outputs.logits.cpu()
        outputs.pred_boxes = outputs.pred_boxes.cpu()
        results = processor.post_process_object_detection_with_nms(outputs=outputs, target_sizes=target_sizes, threshold=self._confidence, nms_threshold=self._nms_threshold)
        # results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

        # Initialize an empty list to store ImageRegion objects
        annotations: list[ImageRegion] = [] 
        # Iterate through the detected boxes, scores, and labels
        for box, score, label in zip(boxes, scores, labels):
            box = [int(i) for i in box.tolist()]
            
            # Skip detections with scores below the threshold
            if score < score_threshold:
                continue
            
            # Unpack the box coordinates
            x1, y1, x2, y2 = box
            
            # Create an ImageRegion object with the detection information
            image_region = ImageRegion(x1, y1, x2, y2, label, score)
            
            # Add the ImageRegion to the annotations list
            annotations.append(image_region)
        
        if self._save_cropped_images:
            self._save_crops(image, annotations, self._output_dir)
    
        return annotations
    
    def _save_crops(self, image: Image, annotations: list[ImageRegion], output_dir: str):
        """
        Saves cropped images based on the provided annotations.

        Args:
            image (Image): The original image to crop.
            annotations (list[ImageRegion]): A list of ImageRegion objects containing the bounding box coordinates.
            output_dir (str): The directory to save the cropped images.
        """
        # Iterate through the annotations
        for i, annotation in enumerate(annotations):
            # Unpack the bounding box coordinates
            x1, y1, x2, y2 = annotation.x1, annotation.y1, annotation.x2, annotation.y2
            
            # Crop the image using the bounding box coordinates
            crop = image.crop((x1, y1, x2, y2))
            
            # Save the cropped image to the output directory
            crop.save(f"{output_dir}/crop_{i}.png")
            
            # Print a message indicating the crop has been saved
            print(f"Cropped image saved to {output_dir}/crop_{i}.png")



FlorenceModelName = Literal[
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large-ft",
    "microsoft/Florence-2-base",
    "microsoft/Florence-2-large"
]

class Florence2ForObjectDetectionModel(ObjectDetectionModel):
    _processor: Optional[AutoProcessor] = None
    _model: Optional[Owlv2ForObjectDetection] = None
    
    def __init__(
            self,
            model_name: FlorenceModelName = "microsoft/Florence-2-base-ft",
            confidence=0.1,
            nms_threshold=0.3,
            save_cropped_images: bool = False,
            output_dir: str = "./output",
            device: str = "cpu",
        ):
        self._model_name = model_name
        self._save_cropped_images = save_cropped_images
        self._output_dir = output_dir
        self._confidence = confidence
        self._nms_threshold = nms_threshold
        self._device = device


    def _get_or_create_florence2(self) -> Owlv2ForObjectDetection:
        """
        Retrieves or creates an instance of the Owlv2ForObjectDetection model.

        Returns:
            Owlv2ForObjectDetection: The object detection model instance.
        """
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name, device_map=self._device, trust_remote_code=True).eval()
        return self._model
    
    def _get_or_create_florence2_processor(self) -> AutoProcessor:
        """
        Retrieves or creates an instance of the Owlv2ProcessorWithNMS processor.

        Returns:
            AutoProcessor: The processor instance for handling image processing.
        """
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self._model_name, trust_remote_code=True)
        return self._processor

    @torch.no_grad()
    def detect_bboxes(self, image_node: ImageDocument, prompt: str, **kwargs) -> list[ImageRegion]:
        """
        Detects bounding boxes in the image using the Owlv2 model.

        Args:
            image_node (ImageDocument): The image node to process.
            prompt (str): The prompt for the object detection model.
            confidence (float): The confidence threshold for detections.
            nms_threshold (float): The non-maximum suppression threshold.

        Returns:
            list[ImageRegion]: A list of detected bounding boxes with their associated labels and scores.
        """
        image = Image.open(image_node.resolve_image()).convert("RGB")
        processor = self._get_or_create_florence2_processor()
        model = self._get_or_create_florence2()

        task_prompt = '<OD>' + prompt

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(self._device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer
    
    def _save_crops(self, image: Image, annotations: list[ImageRegion], output_dir: str):
        """
        Saves cropped images based on the provided annotations.

        Args:
            image (Image): The original image to crop.
            annotations (list[ImageRegion]): A list of ImageRegion objects containing the bounding box coordinates.
            output_dir (str): The directory to save the cropped images.
        """
        # Iterate through the annotations
        for i, annotation in enumerate(annotations):
            # Unpack the bounding box coordinates
            x1, y1, x2, y2 = annotation.x1, annotation.y1, annotation.x2, annotation.y2
            
            # Crop the image using the bounding box coordinates
            crop = image.crop((x1, y1, x2, y2))
            
            # Save the cropped image to the output directory
            crop.save(f"{output_dir}/crop_{i}.png")
            
            # Print a message indicating the crop has been saved
            print(f"Cropped image saved to {output_dir}/crop_{i}.png")