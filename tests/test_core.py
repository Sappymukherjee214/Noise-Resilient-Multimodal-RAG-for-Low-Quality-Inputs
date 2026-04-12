import pytest
from generation.generator import RAGGenerator
from evaluation.metrics import EvaluationSuite
from preprocessing.cleaner import DataCleaner

def test_generator_creation():
    gen = RAGGenerator(provider="openai")
    assert gen is not None
    assert hasattr(gen, 'llm_client')

def test_cleaner_logic():
    cleaner = DataCleaner()
    text = "   MODRIN  blue!! "
    cleaned = cleaner.clean_text(text)
    assert cleaned == "modrin blue"

def test_evaluation_hallucination():
    suite = EvaluationSuite()
    context = "Blue shirt made of silk."
    answer = "The product is a blue shirt."
    eval_res = suite.evaluate_response(context, answer)
    assert eval_res['faithfulness'] > 0.5
    assert eval_res['is_hallucinated'] == False

def test_hallucination_detection():
    suite = EvaluationSuite()
    context = "Blue shirt made of silk."
    answer = "This is a red car."
    eval_res = suite.evaluate_response(context, answer)
    assert eval_res['is_hallucinated'] == True
