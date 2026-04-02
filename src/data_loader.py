import pandas as pd
import kagglehub
import os
from PIL import Image
from typing import Tuple, List, Dict

class FashionDataLoader:
    """Manages the Fashion Product Images (Small) dataset for RAG testing."""

    def __init__(self):
        self.dataset_path = None
        self.styles_df = None
        self.images_dir = None

    def download_and_init(self):
        """Downloads the Kaggle dataset if needed and initializes metadata."""
        import kagglehub
        try:
            self.dataset_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
            print(f"[System]: Dataset located at: {self.dataset_path}")
            
            # Sub-path for standard metadata and images
            csv_path = os.path.join(self.dataset_path, "styles.csv")
            self.images_dir = os.path.join(self.dataset_path, "images")
            
            # Error handling for malformed CSV rows (common in this fashion dataset)
            self.styles_df = pd.read_csv(csv_path, on_bad_lines='skip')
            
            # PRODUCTION-LITE OVERRIDE: Protects free-tier cloud disk space
            if os.environ.get("IS_PRODUCTION", "False").lower() == "true":
                print(f"[Warning]: PRODUCTION_LITE_MODE active. Subsetting dataset to 1,000 items.")
                self.styles_df = self.styles_df.head(1000)
                
            print(f"[System]: Loaded {len(self.styles_df)} product records.")
        except Exception as e:
            print(f"[Error]: Failure in dataset initialization: {e}")

    def get_sample(self, index: int = 0) -> Tuple[str, Image.Image, Dict]:
        """Retrieves a specific sample: (Description, Image, Metadata)."""
        if self.styles_df is None:
            raise ValueError("Dataset not initialized. Call download_and_init() first.")
        
        row = self.styles_df.iloc[index]
        product_id = str(row['id'])
        description = str(row['productDisplayName'])
        
        img_path = os.path.join(self.images_dir, f"{product_id}.jpg")
        
        if not os.path.exists(img_path):
            # Try png if jpg fails
            img_path = os.path.join(self.images_dir, f"{product_id}.png")
            
        image = Image.open(img_path).convert("RGB")
        
        metadata = {
            "id": product_id,
            "category": row['articleType'],
            "base_colour": row['baseColour'],
            "season": row['season']
        }
        
        return description, image, metadata

    def get_random_sample(self) -> Tuple[str, Image.Image, Dict]:
        import random
        idx = random.randint(0, len(self.styles_df) - 1)
        return self.get_sample(idx)

# Simple standalone test
if __name__ == "__main__":
    loader = FashionDataLoader()
    loader.download_and_init()
    desc, img, meta = loader.get_random_sample()
    print(f"Sample Found: {desc}")
    print(f"Metadata: {meta}")
    img.show()
