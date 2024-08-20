from io import BytesIO
from llama_index.core.schema import ImageDocument, ImageNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.workflow import Event,StartEvent,StopEvent,Workflow,step
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from typing import Optional
import base64
import numpy as np
import shutil
import torch