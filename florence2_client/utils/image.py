import io
import os

from PIL import Image


def load_image(image_path: str) -> Image.Image:
    """
    Load and preprocess the image from a given file path.

    Args:
    - image_path (str): Path to the local image file.

    Returns:
    - Image.Image: The processed PIL Image object in RGB format.

    Raises:
    - FileNotFoundError: If the specified image file does not exist.
    - ValueError: If the image cannot be opened or processed.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except Exception as e:
        raise ValueError(f"Unable to load or process the image file: {e}")


def pil_image_to_binary(image: Image.Image) -> io.BytesIO:
    """
    Convert a PIL Image to a binary file-like object with the inferred format.

    Args:
    - image (Image.Image): The input PIL Image object.

    Returns:
    - io.BytesIO: A binary file-like object containing the image data.
    """
    image_buffer = io.BytesIO()

    # Infer format from the PIL image object
    format_inferred = image.format if image.format else "PNG"  # Default to PNG if format is None

    image.save(image_buffer, format=format_inferred)
    image_buffer.seek(0)

    return image_buffer