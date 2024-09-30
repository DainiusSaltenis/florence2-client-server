from PIL import Image
import io
from fastapi import HTTPException, UploadFile


async def load_image(file: UploadFile) -> Image.Image:
    """
    Load and preprocess the image.

    Args:
    - file (UploadFile): File object from FastAPI.

    Returns:
    - Image.Image: PIL Image.

    Raises:
    - HTTPException: If the file is not a valid image or cannot be processed.
    """
    try:
        image = Image.open(io.BytesIO(await file.read()))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
