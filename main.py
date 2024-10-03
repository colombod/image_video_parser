from llama_index.core.schema import ImageNode
from PIL import Image
import shutil
import asyncio

from src.image_node_parser_workflow import ImageNodeParserWorkflow
from llama_index.utils.workflow import draw_all_possible_flows, draw_most_recent_execution
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal 
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    sam_config = {
        "model_name": "facebook/sam2-hiera-small",
        # "prompt": "all the lamps",
    }
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

    # Draw all
    # draw_all_possible_flows(
    #     workflow,
    #     filename="workflow.html"
    # )
    
    result = await workflow.run(image_path="./images/Diablo® IV-2024_10_01-15_03_15.png", segmentation_configuration=sam_config)
    
    # Draw an execution
    # draw_most_recent_execution(result, filename="workflow_run.html")

    # remove the ./output folder
    shutil.rmtree("./output", ignore_errors=True)
    shutil.os.mkdir("./output")

    for chunk in result["chunks"]:
        if isinstance(chunk, ImageNode):
            Image.open(chunk.resolve_image()).save(f"./output/{chunk.node_id}.png")

if __name__ == "__main__":
    asyncio.run(main())
