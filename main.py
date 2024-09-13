from llama_index.core.schema import ImageNode
from PIL import Image
import shutil
import asyncio

from src.image_node_parser_workflow import ImageNodeParserWorkflow
from llama_index.utils.workflow import draw_all_possible_flows, draw_most_recent_execution
 

async def main():
    sam_config = {
        "model_name": "facebook/sam2-hiera-small",
        "prompt": "all the lamps",
    }   

    workflow = ImageNodeParserWorkflow(verbose=True)

    # Draw all
    draw_all_possible_flows(
        workflow,
        filename="workflow.html"
    )
    
    result = await workflow.run(image_path="./images/ikea.png", segmentation_configuration=sam_config)
    
    # Draw an execution
    draw_most_recent_execution(result, filename="workflow_run.html")

    # remove the ./output folder
    shutil.rmtree("./output", ignore_errors=True)
    shutil.os.mkdir("./output")

    for chunk in result["chunks"]:
        if isinstance(chunk, ImageNode):
            Image.open(chunk.resolve_image()).save(f"./output/{chunk.node_id}.png")

if __name__ == "__main__":
    asyncio.run(main())
