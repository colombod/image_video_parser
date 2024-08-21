from llama_index.core.schema import ImageNode
from PIL import Image
import shutil
import asyncio

from src.image_node_parser_workflow import ImageNodeParserWorkflow

async def main():
    sam_config = {
        "model_name": "facebook/sam2-hiera-small",
        "settings": {}
    }   

    workflow = ImageNodeParserWorkflow(verbose=True)
    result = await workflow.run(image_path="./images/il_vulcano_2.png", segmentation_configuration=sam_config)

    # remove the ./output folder
    shutil.rmtree("./output", ignore_errors=True)
    shutil.os.mkdir("./output")

    for chunk in result["chunks"]:
        if isinstance(chunk, ImageNode):
            Image.open(chunk.resolve_image()).save(f"./output/{chunk.node_id}.png")

if __name__ == "__main__":
    asyncio.run(main())
