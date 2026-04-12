import re
from typing import Union, Any

class DataCleaner:
    """Provides robust cleaning logic for noisy multimodal inputs."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalizes text, removes artifacts, and corrects basic spacing.
        (Industry practice: Use a lightweight normalization pipeline)
        """
        if not text:
            return ""
        
        # Lowercase and remove special chars
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Note: In a full production system, we'd use a spellchecker like 'pyspellchecker'
        # or an LLM-based refiner. For this modular build, we keep it core/fast.
        return text

    @staticmethod
    def denoise_image(image: Any) -> Any:
        """
        Applies Non-Local Means Denoising to remove sensor noise and compression artifacts.
        """
        import cv2
        import numpy as np
        from PIL import Image

        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image

        # OpenCV expects BGR for most functions. Convert RGB to BGR.
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # OpenCV fastNlMeansDenoisingColored
        denoised_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
        
        # Convert back to RGB for PIL consistency
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(denoised_rgb)

    @staticmethod
    def resize_for_vlm(image: Any, size: tuple = (224, 224)) -> Any:
        """Standardizes image size for VLM encoders like CLIP."""
        from PIL import Image
        return image.resize(size, Image.Resampling.LANCZOS)

if __name__ == "__main__":
    cleaner = DataCleaner()
    print(cleaner.clean_text("Modrin  fashion!!!   item"))
