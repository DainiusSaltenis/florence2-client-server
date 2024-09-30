import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def load_florence(model_name: str, device: str, torch_dtype: torch.dtype):
    """
    Load Florence-2 model and processor.

    Args:
    - model_name (str): HuggingFace model identifier.
    - device (str): Torch device to load the model on.
    - torch_dtype (torch.dtype): Torch data type for the model weights.

    Returns:
    - model: Loaded model object.
    - processor: Corresponding image processor.
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    return model, processor


def inference(model, processor, image, task_prompt: str, device: str, torch_dtype: torch.dtype, text_input: str = None):
    """
    Run inference using Florence-2.

    Args:
    - model: The loaded model object of Florence-2.
    - processor: The processor object.
    - image: The preprocessed input image.
    - task_prompt (str): The task text prompt.
    - device (str): Torch device to run inference on.
    - torch_dtype (torch.dtype): Torch data type for input tensor.
    - text_input (Optional[str], default=None): Text to add to the task prompt.

    Returns:
    - generated_ids: The generated token IDs from the model.
    - response: Processed model output.
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Prepare inputs for the model
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    # Generate caption
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=50,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    response = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return generated_text, response
