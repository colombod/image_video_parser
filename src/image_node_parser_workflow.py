from typing import Optional
from llama_index.core.schema import ImageDocument, ImageNode, NodeRelationship, RelatedNodeInfo
from PIL import Image
from io import BytesIO
import shutil
import base64
import torch
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator