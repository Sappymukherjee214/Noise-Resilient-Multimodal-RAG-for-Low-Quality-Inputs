from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from evaluation.hallucination import HallucinationDetector

class EvaluationSuite:
    """Production-grade evaluation framework for RAG quality and reliability."""

    def __init__(self, log_dir: str = "logs/eval_records"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.smooth = SmoothingFunction().method1
        self.hallucination_detector = HallucinationDetector()

    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculates BLEU-4 score."""
        if not reference or not candidate:
            return 0.0
        ref_tokens = [reference.lower().split()]
        cand_tokens = candidate.lower().split()
        return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=self.smooth)

    def evaluate_response(self, context: str, answer: str, expected: str = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single response.
        """
        hallucination_results = self.hallucination_detector.check_hallucination(context, answer)
        
        metrics = {
            "hallucination_score": hallucination_results["score"],
            "faithfulness": hallucination_results["grounding_score"],
            "is_hallucinated": hallucination_results["is_hallucinated"],
            "reason": hallucination_results["reason"]
        }

        if expected:
            metrics["bleu"] = self.calculate_bleu(expected, answer)
            
        return metrics

    def generate_benchmark_report(self, baseline_results: List[Dict], optimized_results: List[Dict]) -> Dict[str, Any]:
        """
        Generates a side-by-side comparison report between two system versions.
        """
        def summarize(results):
            if not results: return {"bleu": 0, "faith": 0}
            return {
                "avg_bleu": sum(r.get('bleu', 0) for r in results) / len(results),
                "avg_faithfulness": sum(r.get('faithfulness', 0) for r in results) / len(results),
                "hallucination_rate": sum(1 for r in results if r.get('is_hallucinated')) / len(results)
            }

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "baseline": summarize(baseline_results),
            "optimized": summarize(optimized_results),
            "delta": {}
        }

        # Calculate deltas
        for key in comparison["baseline"]:
            comparison["delta"][key] = comparison["optimized"][key] - comparison["baseline"][key]

        self.save_results(comparison, "benchmark_report")
        return comparison

    def evaluate_batch(self, test_set: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a batch of RAG outputs with deep metrics.
        """
        processed_results = []
        for test, res in zip(test_set, results):
            metrics = self.evaluate_response(
                context=res.get('context', ''),
                answer=res.get('answer', ''),
                expected=test.get('expected_output')
            )
            processed_results.append({
                "id": test.get('id'),
                "query": test.get('query'),
                "metrics": metrics
            })
            
        summary = {
            "timestamp": datetime.now().isoformat(),
            "avg_bleu": sum(r['metrics'].get('bleu', 0) for r in processed_results) / len(processed_results),
            "avg_faithfulness": sum(r['metrics']['faithfulness'] for r in processed_results) / len(processed_results),
            "hallucination_rate": sum(1 for r in processed_results if r['metrics']['is_hallucinated']) / len(processed_results),
            "details": processed_results
        }
        
        self.save_results(summary, "batch_eval")
        return summary

    def save_results(self, results: Dict[str, Any], prefix: str):
        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(self.log_dir, filename), "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    suite = EvaluationSuite()
    test_context = "This is a blue shirt made of cotton."
    test_answer = "The product is a blue shirt."
    print(suite.evaluate_response(test_context, test_answer))

