import requests
import supervision as sv
from PIL import Image


class Florence2Client:
    def __init__(self, base_url: str):
        """
        Initialize the Florence-2 HTTP client.

        Args:
        - base_url (str): Base URL of the API inference server.
        """
        self.base_url = base_url

    def _send_request(self, endpoint: str, image_path: str, params: dict = None) -> dict:
        """
        Send image requests to the Florence-2 inference server.

        Args:
        - endpoint (str): API endpoint, e.g., '/caption'.
        - image_path (str): Path to the image file stored locally.
        - params (dict): Additional query parameters.

        Returns:
        - dict: JSON response from the server.
        """
        with open(image_path, "rb") as image_file:
            response = requests.post(f"{self.base_url}{endpoint}", files={"file": image_file}, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def generate_caption(self, image_path: str, caption_verbosity: str = "default") -> str:
        """
        Generate a caption for the given image.

        Args:
        - image_path (str): Path to the local image file.
        - caption_verbosity (str): Verbosity level. Options: 'default', 'detailed', 'more_detailed'.

        Returns:
        - str: The generated caption.
        """
        result = self._send_request("/caption", image_path, {"caption_verbosity": caption_verbosity})
        return result.get("caption", "")

    def detect_objects(self, image_path: str) -> sv.Detections:
        """
        Perform object detection on the given image and return formatted detections.

        Args:
        - image_path (str): Path to the local image file.

        Returns:
        - sv.Detections: Detections formatted for supervision integration.
        """
        result = self._send_request("/detect-objects", image_path)

        # Convert detection results to supervision's Detections format
        detections = sv.Detections.from_json(result["detections"])
        return detections

    def visualize_detections(self, image_path: str, detections: sv.Detections) -> Image.Image:
        """
        Visualize object detections using Roboflow Supervision.

        Args:
        - image_path (str): Path to the local image file.
        - detections (sv.Detections): Detected objects in the image.

        Returns:
        - Image.Image: The image with bounding boxes drawn.
        """
        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Create a detection overlay using Roboflow Supervision
        overlay = sv.draw_detections(image, detections)
        return overlay