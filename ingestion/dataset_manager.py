import pandas as pd
import kagglehub
import os
import json
import logging
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional
from config.settings import settings

logger = logging.getLogger(__name__)

class DatasetManager:
    """Production-grade Dataset Manager with versioning and quality governance."""

    REQUIRED_COLUMNS = ['id', 'productDisplayName', 'baseColour']

    def __init__(self):
        self.dataset_path = None
        self.metadata_df = None
        self.images_dir = None
        self.version_info = {
            "version": settings.DATASET_VERSION,
            "source": settings.KAGGLE_DATASET,
            "processed_steps": [],
            "quality_summary": {}
        }

    def download_and_init(self) -> str:
        """Downloads the dataset and performs governance checks."""
        try:
            logger.info(f"Downloading {settings.KAGGLE_DATASET}...")
            self.dataset_path = kagglehub.dataset_download(settings.KAGGLE_DATASET)
            
            csv_path = os.path.join(self.dataset_path, "styles.csv")
            self.images_dir = os.path.join(self.dataset_path, "images")
            
            # Load metadata
            self.metadata_df = pd.read_csv(csv_path, on_bad_lines='skip')
            
            # 1. Schema Validation
            self._validate_schema()
            
            # 2. Quality Scoring
            self._score_data_quality()
                
            self._save_version_lock()
            logger.info(f"Dataset initialized with {len(self.metadata_df)} records.")
            return self.dataset_path
        except Exception as e:
            logger.error(f"Dataset Initialization Error: {e}")
            raise

    def _validate_schema(self):
        """Ensures required columns exist."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.metadata_df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

    def _score_data_quality(self):
        """Calculates completeness and quality metrics."""
        null_counts = self.metadata_df[self.REQUIRED_COLUMNS].isnull().sum()
        completeness = 1.0 - (null_counts.sum() / (len(self.metadata_df) * len(self.REQUIRED_COLUMNS)))
        
        self.version_info["quality_summary"] = {
            "completeness_score": float(completeness),
            "total_records": len(self.metadata_df),
            "missing_ids": int(null_counts['id'])
        }

    def get_quality_sample(self, idx: int) -> Dict[str, Any]:
        """Returns sample with data quality metadata."""
        base_sample = self.get_sample(idx)
        
        # Calculate sample-specific quality
        quality_score = 1.0
        if not os.path.exists(base_sample['image_path']):
            quality_score -= 0.5
        if not base_sample['text'] or base_sample['text'] == "nan":
            quality_score -= 0.5
            
        base_sample["quality_score"] = max(0.0, quality_score)
        return base_sample

    def get_sample(self, idx: int) -> Dict[str, Any]:
        """Returns a structured data sample."""
        if self.metadata_df is None:
            raise ValueError("Dataset not initialized.")
            
        row = self.metadata_df.iloc[idx]
        product_id = str(row['id'])
        img_path = os.path.join(self.images_dir, f"{product_id}.jpg")
        
        return {
            "id": product_id,
            "text": str(row.get('productDisplayName', 'N/A')),
            "image_path": img_path,
            "metadata": {
                "category": row.get('articleType', 'N/A'),
                "base_colour": row.get('baseColour', 'N/A'),
                "gender": row.get('gender', 'N/A')
            }
        }

    def _save_version_lock(self):
        lock_path = os.path.join(settings.DATA_DIR, "dataset_lock.json")
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        with open(lock_path, "w") as f:
            json.dump(self.version_info, f, indent=4)


if __name__ == "__main__":
    dm = DatasetManager()
    dm.download_and_init()
    print(dm.get_sample(0))
