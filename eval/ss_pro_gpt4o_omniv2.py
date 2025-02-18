import os
import re
import ast
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import openai
from openai import BadRequestError

model_name = "gpt-4o-2024-05-13"
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


from models.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model,
)
import torch
from ultralytics import YOLO
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
SOM_MODEL_PATH = "..."
CAPTION_MODEL_PATH = "..."
som_model = get_yolo_model(SOM_MODEL_PATH)

som_model.to(device)
print("model to {}".format(device))

# two choices for caption model: fine-tuned blip2 or florence2
import importlib

caption_model_processor = get_caption_model_processor(
    model_name="florence2", model_name_or_path="CAPTION_MODEL_PATH", device=device
)


def omniparser_parse(image, image_path):
    raise NotImplementedError("omniparser_parse method is not implemented yet.")


def reformat_messages(parsed_content_list):
    raise NotImplementedError("reformat_messages method is not implemented yet.")


PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT = """Please generate the next move according to the UI screenshot and task instruction. You will be presented with a screenshot image. Also you will be given each bounding box's description in a list. To complete the task, You should choose a related bbox to click based on the bbox descriptions.
Task instruction: {}.
Here is the list of all detected bounding boxes by IDs and their descriptions: {}. Keep in mind the description for Text Boxes are likely more accurate than the description for Icon Boxes.
Requirement: 1. You should first give a reasonable description of the current screenshot, and give a short analysis of how can the user task be achieved. 2. Then make an educated guess of bbox id to click in order to complete the task based on the bounding boxes descriptions. 3. Your answer should follow the following format: {{"Analysis": xxx, "Click BBox ID": "y"}}. Do not include any other info. Some examples: {}. The task is to {}. Retrieve the bbox id where its description matches the task instruction. Now start your answer:"""

# PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT_v1 = "The instruction is to {}. \nHere is the list of all detected bounding boxes by IDs and their descriptions: {}. \nKeep in mind the description for Text Boxes are likely more accurate than the description for Icon Boxes. \n Requirement: 1. You should first give a reasonable description of the current screenshot, and give a step by step analysis of how can the user task be achieved. 2. Then make an educated guess of bbox id to click in order to complete the task using both the visual information from the screenshot image and the bounding boxes descriptions. 3. Your answer should follow the following format: {{'Analysis': 'xxx', 'Click BBox ID': 'y'}}. Please do not include any other info."
PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT_v1 = "The instruction is to {}. \nHere is the list of all detected bounding boxes by IDs and their descriptions: {}. \nKeep in mind the description for Text Boxes are likely more accurate than the description for Icon Boxes. \n Requirement: 1. You should first give a reasonable description of the current screenshot, and give a some analysis of how can the user instruction be achieved by a single click. 2. Then make an educated guess of bbox id to click in order to complete the task using both the visual information from the screenshot image and the bounding boxes descriptions. REMEMBER: the task instruction must be achieved by one single click. 3. Your answer should follow the following format: {{'Analysis': 'xxx', 'Click BBox ID': 'y'}}. Please do not include any other info."


FEWSHOT_EXAMPLE = """Example 1: Task instruction: Next page. \n{"Analysis": "Based on the screenshot and icon descriptions, I should click on the next page icon, which is labeled with box ID x in the bounding box list", "Click BBox ID": "x"}\n\n
Example 2: Task instruction: Search on google. \n{"Analysis": "Based on the screenshot and icon descriptions, I should click on the 'Search' box, which is labeled with box ID y in the bounding box list", "Click BBox ID": "y"}"""


from azure.identity import (
    AzureCliCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI
from models.utils import get_pred_phi3v, extract_dict_from_text, get_phi3v_model_dict


class GPT4XModel:
    def __init__(self, model_name="gpt-4o-2024-05-13", use_managed_identity=False):
        raise NotImplementedError("GPT4XModel.__init__ method is not implemented yet.")

    def load_model(self):
        raise NotImplementedError("load_model method is not implemented yet.")

    def set_generation_config(self, **kwargs):
        raise NotImplementedError(
            "set_generation_config method is not implemented yet."
        )

    def ground_only_positive_phi35v(self, instruction, image):
        raise NotImplementedError(
            "ground_only_positive_phi35v method is not implemented yet."
        )

    def ground_only_positive(self, instruction, image):
        raise NotImplementedError("ground_only_positive method is not implemented yet.")

    def ground_allow_negative(self, instruction, image=None):
        raise NotImplementedError(
            "ground_allow_negative method is not implemented yet."
        )

    def ground_with_uncertainty(self, instruction, image=None):
        raise NotImplementedError(
            "ground_with_uncertainty method is not implemented yet."
        )


def extract_first_bounding_box(text):
    raise NotImplementedError(
        "extract_first_bounding_box method is not implemented yet."
    )


def extract_first_point(text):
    raise NotImplementedError("extract_first_point method is not implemented yet.")
