from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import requests
import torch

from utils.image import load_image
from utils.model import load_florence, inference

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# TODO: Selectable model type
DEFAULT_MODEL = "microsoft/Florence-2-base-ft"

app = FastAPI()
model, processor = load_florence(DEFAULT_MODEL, DEVICE, TORCH_DTYPE)


@app.on_event("startup")
async def load_model():
    """
    Load and cache the Florence-2 model during startup.
    """
    # Dummy input to ensure the model is loaded and cached correctly
    dummy_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(dummy_image_url, stream=True).raw)
    dummy_inputs = processor(text="<OD>", images=image, return_tensors="pt").to(DEVICE, TORCH_DTYPE)

    _ = model.generate(
        input_ids=dummy_inputs["input_ids"],
        pixel_values=dummy_inputs["pixel_values"],
        max_new_tokens=10,
        do_sample=False,
        num_beams=1,
    )


@app.get("/")
async def root():
    return {"message": "Florence-2 inference API is running."}


@app.post("/caption")
async def generate_caption(file: UploadFile = File(...), caption_verbosity: str = "default"):
    """
    Generate a caption for an uploaded image.
    Args:
    - file (UploadFile): The image file to generate a caption for.
    - verbosity (str): Can be 'normal', 'detailed', or 'more_detailed'.

    Returns:
    - JSON containing the generated caption.
    """
    # Set prompt based on verbosity
    if caption_verbosity == "detailed":
        task_prompt = "<DETAILED_CAPTION>"
    elif caption_verbosity == "more_detailed":
        task_prompt = "<MORE_DETAILED_CAPTION>"
    else:
        task_prompt = "<CAPTION>"

    image = await load_image(file)

    generated_text, _ = inference(model, processor, image, task_prompt, DEVICE, TORCH_DTYPE)

    return {"caption": generated_text}


@app.post("/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    """
    Perform object detection on an uploaded image.
    Args:
    - file (UploadFile): The image file to analyze.

    Returns:
    - JSON containing detected objects and their bounding boxes.
    """
    task_prompt = "<OD>"
    image = await load_image(file)

    _, response = inference(model, processor, image, task_prompt, DEVICE, TORCH_DTYPE)

    return {"detections": response}


@app.post("/detect-regions-with-captions")
async def detect_regions_with_captions(file: UploadFile = File(...)):
    """
    Perform dense region caption detection on an uploaded image.
    Args:
    - file (UploadFile): The image file to analyze.

    Returns:
    - JSON containing dense captions with detected regions.
    """
    task_prompt = "<DENSE_REGION_CAPTION>"
    image = await load_image(file)

    _, response = inference(model, processor, image, task_prompt, DEVICE, TORCH_DTYPE)

    return {"regions_with_captions": response}


@app.post("/detect-region-proposals")
async def detect_region_proposals(file: UploadFile = File(...)):
    """
    Perform region proposals on an uploaded image.
    Args:
    - file (UploadFile): The image file to analyze.

    Returns:
    - JSON containing region proposals.
    """
    task_prompt = "<REGION_PROPOSAL>"
    image = await load_image(file)

    _, response = inference(model, processor, image, task_prompt, DEVICE, TORCH_DTYPE)

    return {"region_proposals": response}


@app.post("/open-vocabulary-detection")
async def open_vocabulary_detection(file: UploadFile = File(...), class_name: str = ""):
    """
    Perform open vocabulary object detection.
    Args:
    - file (UploadFile): The image file to analyze.
    - class_name (str): Which class object to detect.

    Returns:
    - JSON containing captions with detected regions.
    """
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    image = await load_image(file)

    _, response = inference(model, processor, image, task_prompt, DEVICE, TORCH_DTYPE, text_input=class_name)

    return {"detections": response}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    Perform optical character recognition.
    Args:
    - file (UploadFile): The image file to analyze.

    Returns:
    - JSON containing text strings with corresponding bounding boxes and strings.
    """
    task_prompt = "<OCR_WITH_REGION>"
    image = await load_image(file)

    _, response = inference(model, processor, image, task_prompt, DEVICE, TORCH_DTYPE)

    return {"ocr": response}


@app.post("/referring-expression-segmentation")
async def referring_expression_segmentation(file: UploadFile = File(...), expression: str = ""):
    """
    Perform referring expression segmentation.
    Args:
    - file (UploadFile): The image file to analyze.
    - expression (str): Expression describing the class to segment.

    Returns:
    - JSON containing text strings with corresponding segmentation masks.
    """
    task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
    image = await load_image(file)

    _, response = inference(model, processor, image, task_prompt, DEVICE, TORCH_DTYPE, text_input=expression)

    return {"segmentation": response}


# TODO: <REGION_TO_SEGMENTATION>
# TODO: <REGION_TO_CATEGORY>
# TODO: <REGION_TO_DESCRIPTION>
# TODO: <CAPTION_TO_PHRASE_GROUNDING>
# TODO: Add a method to manage cached model, allow changing versions
