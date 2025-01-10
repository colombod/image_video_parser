from llama_index.core.schema import ImageDocument
from PIL import Image
import shutil
import asyncio

from image_video_parser.image_node_parser_workflow import ImageNodeParserWorkflow
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from dotenv import load_dotenv
import os

from image_video_parser.object_detection_model import Florence2ForObjectDetectionModel, OwlV2ObjectDetectionModel
from image_video_parser.object_segmentation_model import SamForImageSegmentation

load_dotenv()

async def main():
    # remove the ./output folder
    shutil.rmtree("./output", ignore_errors=True)
    shutil.os.mkdir("./output")
    shutil.os.mkdir("./output/cropped_images")
    shutil.os.mkdir("./output/segmented_images")
    
    azure_openai_mm_llm = OpenAIMultiModal(
        model=os.getenv("MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_new_tokens=300,
    )

    workflow = ImageNodeParserWorkflow(verbose=True)
    workflow.multi_modal_llm = azure_openai_mm_llm
    # workflow.object_detection_model = Florence2ForObjectDetectionModel(save_cropped_images=True)
    workflow.object_detection_model = OwlV2ObjectDetectionModel(save_cropped_images=True, output_dir="./output/cropped_images")
    workflow.image_segmentation_model = SamForImageSegmentation(model_name="facebook/sam2.1-hiera-large", device="cpu")
    
    # result = await workflow.run(image_path="./images/ikea.png", prompt="all the chairs")
    result = await workflow.run(image_path="./images/diablo_menu.png")

    if isinstance(result, str):
        print(result)
        return

    for chunk in result["chunks"]:
        if isinstance(chunk, ImageDocument):
            Image.open(chunk.resolve_image()).save(f"./output/segmented_images/{chunk.node_id}.png")

if __name__ == "__main__":
    asyncio.run(main())
