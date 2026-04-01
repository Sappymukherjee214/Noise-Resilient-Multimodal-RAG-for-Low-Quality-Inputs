import numpy as np
import cv2
from PIL import Image, ImageFilter
import random
import nltk
from typing import Union, List

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextNoiseInjector:
    """Simulates realistic text degradation."""
    
    @staticmethod
    def char_typos(text: str, probability: float = 0.1) -> str:
        """Injects character-level insertion, deletion, and substitution."""
        chars = list(text)
        new_chars = []
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        
        for char in chars:
            if random.random() < probability:
                noise_type = random.choice(['del', 'ins', 'sub'])
                if noise_type == 'del':
                    continue
                elif noise_type == 'ins':
                    new_chars.append(char)
                    new_chars.append(random.choice(alphabet))
                else: # sub
                    new_chars.append(random.choice(alphabet))
            else:
                new_chars.append(char)
        return "".join(new_chars)

    @staticmethod
    def semantic_swap(text: str, probability: float = 0.05) -> str:
        """Informalizes or swaps words (placeholder for complex semantic noise)."""
        words = text.split()
        for i in range(len(words)):
            if random.random() < probability:
                # Simplistic informal replacement
                if words[i] in ["you", "your"]:
                    words[i] = "u"
                elif words[i] == "are":
                    words[i] = "r"
        return " ".join(words)

class ImageNoiseInjector:
    """Simulates visual artifacts typical in low-quality captures."""
    
    @staticmethod
    def add_gaussian_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def add_salt_and_pepper(image: Image.Image, amount: float = 0.02) -> Image.Image:
        """Adds salt and pepper noise using numpy and cv2."""
        img_np = np.array(image)
        noise = np.random.rand(*img_np.shape[:2])
        img_np[noise < amount / 2] = 0
        img_np[noise > 1 - amount / 2] = 255
        return Image.fromarray(img_np)

    @staticmethod
    def simulate_compression(image: Image.Image, quality: int = 10) -> Image.Image:
        """Simulates JPEG compression artifacts."""
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class AdversarialNoiseInjector:
    """Simulates targeted perturbations to fool multimodal encoders."""
    
    @staticmethod
    def semantic_adversarial_shift(text: str) -> str:
        """Injects non-random but confusing phonetic substitutions."""
        # Concept: Replace key identifying syllables with common homophones
        replacements = {"modern": "modrin", "park": "paark", "bench": "binch"}
        words = text.split()
        for i in range(len(words)):
            low = words[i].lower()
            if low in replacements:
                words[i] = replacements[low]
        return " ".join(words)

def apply_multimodal_noise(text: str, image: Image.Image, noise_level: float = 0.2, 
                            is_adversarial: bool = False):
    """Integrative noise application with optional adversarial mode."""
    t_noise = TextNoiseInjector()
    i_noise = ImageNoiseInjector()
    
    if is_adversarial:
        noisy_text = AdversarialNoiseInjector.semantic_adversarial_shift(text)
    else:
        noisy_text = t_noise.char_typos(text, probability=noise_level)
    
    noisy_image = i_noise.add_gaussian_blur(image, radius=noise_level * 5)
    
    return noisy_text, noisy_image
