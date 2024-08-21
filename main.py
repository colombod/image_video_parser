from io import BytesIO
from llama_index.core.schema import ImageDocument, ImageNode
from PIL import Image
import base64
import shutil
import asyncio

from src.image_node_parser_workflow import ImageNodeParserWorkflow

# image = Image.open("./images/il_vulcano_3.png").convert("RGB")

# def image_to_base64(pil_image, format="JPEG"):
#     buffered = BytesIO()
#     pil_image.save(buffered, format=format)
#     image_str = base64.b64encode(buffered.getvalue())
#     return image_str.decode('utf-8') # Convert bytes to string

# mimetype = "image/jpg"

# document = ImageDocument(image=image_to_base64(image), mimetype=mimetype, image_mimetype=mimetype)

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
