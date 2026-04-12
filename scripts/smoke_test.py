
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from preprocessing.cleaner import DataCleaner
from generation.prompt_manager import PromptManager
from evaluation.hallucination import HallucinationDetector
from ingestion.dataset_manager import DatasetManager

def test_pipeline_logic():
    print("--- Starting Pipeline Logic Smoke Test ---")
    
    # 1. Test Cleaner
    cleaner = DataCleaner()
    input_text = "  MODRIN   blue!! fashion"
    cleaned_text = cleaner.clean_text(input_text)
    print(f"[Cleaner] Input: '{input_text}' -> Output: '{cleaned_text}'")
    assert cleaned_text == "modrin blue fashion"
    
    # 2. Test Prompt Manager with new resilient formatting
    pm = PromptManager()
    prompt = pm.get_prompt("v2_robust", query=cleaned_text) # missing 'confidence' and 'context'
    print(f"[PromptManager] Resilient formatting test (missing keys):")
    print(f"   -> {prompt['user'][:100]}...")
    assert "{query}" not in prompt['user']
    assert "modrin blue fashion" in prompt['user']
    
    # 3. Test Hallucination Detector with new logic
    hd = HallucinationDetector()
    context = "Blue shirt made of cotton."
    answer = "The product is a blue shirt."
    eval_res = hd.check_hallucination(context, answer)
    print(f"[Hallucination] Context: '{context}' | Answer: '{answer}'")
    print(f"   -> Faithfulness: {eval_res['grounding_score']} | Hallucinated: {eval_res['is_hallucinated']}")
    assert eval_res['grounding_score'] > 0.5 # Should be 1.0 (blue, shirt) / 2 words = 1.0 (excluding stop words)
    
    # 4. Test DatasetManager with new metadata keys (Standardized)
    # Mocking the row data to avoid loading 1GB CSV
    mock_row = {
        'id': '123',
        'productDisplayName': 'Mock Blue Shirt',
        'baseColour': 'Blue',
        'articleType': 'Shirt',
        'gender': 'Men'
    }
    dm = DatasetManager()
    dm.metadata_df = MagicMock() # To avoid actual CSV load error
    dm.metadata_df.iloc = [mock_row]
    dm.images_dir = "mock_images"
    
    sample = dm.get_sample(0)
    print(f"[DatasetManager] Metadata key check:")
    print(f"   -> Keys found: {list(sample['metadata'].keys())}")
    assert "base_colour" in sample['metadata']
    assert sample['metadata']['base_colour'] == "Blue"
    
    print("--- Smoke Test Completed Successfully! ---")

if __name__ == "__main__":
    try:
        test_pipeline_logic()
    except Exception as e:
        print(f"Smoke Test Failed: {e}")
        sys.exit(1)
