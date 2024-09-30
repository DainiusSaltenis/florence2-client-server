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
    # TODO: Eliminate caching and store locally / optional download if local not found
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

    image = load_image(file)

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
    image = await load_image(file)

    task_prompt = "<OD>"

    _, response = inference(model, processor, image, task_prompt, DEVICE, TORCH_DTYPE)

    return {"detections": response}
