from llama_index.core.schema import ImageNode
from PIL import Image
import shutil
import asyncio

from src.image_node_parser_workflow import ImageNodeParserWorkflow
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal 
from dotenv import load_dotenv
import os

from src.object_detection_model import Florence2ForObjectDetectionModel, OwlV2ObjectDetectionModel
from src.object_segmentation_model import SamForImageSegmentation

load_dotenv()

async def main():
    # remove the ./output folder
    shutil.rmtree("./output", ignore_errors=True)
    shutil.os.mkdir("./output")
    
    azure_openai_mm_llm = AzureOpenAIMultiModal(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        engine=os.getenv("MODEL"),
        model=os.getenv("MODEL"),
        api_version=os.getenv("API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        max_new_tokens=300,
    )

    workflow = ImageNodeParserWorkflow(verbose=True)
    workflow.multi_modal_llm = azure_openai_mm_llm
    # workflow.object_detection_model = Florence2ForObjectDetectionModel(save_cropped_images=True)
    workflow.object_detection_model = OwlV2ObjectDetectionModel(save_cropped_images=True)
    workflow.image_segmentation_model = SamForImageSegmentation(model_name="facebook/sam2-hiera-small")
    
    result = await workflow.run(image_path="./images/ikea.png", prompt="all chairs")

    if isinstance(result, str):
        print(result)
        return

    for chunk in result["chunks"]:
        if isinstance(chunk, ImageNode):
            Image.open(chunk.resolve_image()).save(f"./output/{chunk.node_id}.png")

if __name__ == "__main__":
    asyncio.run(main())
