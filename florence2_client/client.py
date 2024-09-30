import requests
import supervision as sv
from PIL import Image

from florence2_client.utils.image import pil_image_to_binary, load_image


class Florence2Client:
    def __init__(self, base_url: str):
        """
        Initialize the Florence-2 HTTP client.

        Args:
        - base_url (str): Base URL of the API inference server.
        """
        self.base_url = base_url

    def _send_request(self, endpoint: str, image: Image.Image, params: dict = None) -> dict:
        """
        Send image requests to the Florence-2 inference server.

        Args:
        - endpoint (str): API endpoint, e.g., '/caption'.
        - image_path (str): Path to the image file stored locally.
        - params (dict): Additional query parameters.

        Returns:
        - dict: JSON response from the server.
        """
        image_binary = pil_image_to_binary(image)
        response = requests.post(f"{self.base_url}{endpoint}", files={"file": image_binary}, params=params)

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
        image = load_image(image_path)
        result = self._send_request("/caption", image, {"caption_verbosity": caption_verbosity})
        return result.get("caption", "")

    def detect_objects(self, image_path: str) -> sv.Detections:
        """
        Perform object detection on the given image and return formatted detections.

        Args:
        - image_path (str): Path to the local image file.

        Returns:
        - sv.Detections: Detections formatted for supervision integration.
        """
        image = load_image(image_path)
        result = self._send_request("/detect-objects", image)

        # Convert detection results to supervision's Detections format
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, result.get("detections", ""), resolution_wh=image.size)
        return detections

    def detect_regions_with_captions(self, image_path: str) -> sv.Detections:
        """
        Perform dense region caption detection on an uploaded image.

        Args:
        - image_path (str): Path to the local image file.

        Returns:
        - sv.Detections: Detections formatted for supervision integration.
        """
        image = load_image(image_path)
        result = self._send_request("/detect-regions-with-captions", image)

        # Convert detection results to supervision's Detections format
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, result.get("regions_with_captions", ""), resolution_wh=image.size)
        return detections

    def detect_region_proposals(self, image_path: str) -> sv.Detections:
        """
        Perform region proposal detection on an uploaded image.

        Args:
        - image_path (str): Path to the local image file.

        Returns:
        - sv.Detections: Detections formatted for supervision integration.
        """
        image = load_image(image_path)
        result = self._send_request("/detect-region-proposals", image)

        # Convert detection results to supervision's Detections format
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, result.get("region_proposals", ""), resolution_wh=image.size)
        return detections

    def open_vocabulary_detection(self, image_path: str, class_name: str) -> sv.Detections:
        """
        Perform open vocabulary detection on an uploaded image.

        Args:
        - image_path (str): Path to the local image file.
        - class_name (str): Which class object to detect.

        Returns:
        - sv.Detections: Detections formatted for supervision integration.
        """
        image = load_image(image_path)
        result = self._send_request("/open-vocabulary-detection", image, {"class_name": class_name})

        # Convert detection results to supervision's Detections format
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, result.get("detections", ""), resolution_wh=image.size)
        return detections

    def ocr(self, image_path: str) -> sv.Detections:
        """
        Perform ocr on an uploaded image.

        Args:
        - image_path (str): Path to the local image file.

        Returns:
        - sv.Detections: Detections formatted for supervision integration.
        """
        image = load_image(image_path)
        result = self._send_request("/ocr", image)

        # Convert detection results to supervision's Detections format
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, result.get("ocr", ""), resolution_wh=image.size)
        return detections

    def referring_expression_segmentation(self, image_path: str, expression: str) -> sv.Detections:
        """
        Perform referring expression segmentation on an uploaded image.

        Args:
        - image_path (str): Path to the local image file.

        Returns:
        - sv.Detections: Segmentation masks formatted for supervision integration.
        """
        image = load_image(image_path)
        result = self._send_request("/referring-expression-segmentation", image, {"expression": expression})

        # Convert segmentation results to supervision's Detections format
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, result.get("segmentation", ""), resolution_wh=image.size)
        return detections
