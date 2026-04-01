import numpy as np
from scipy import stats

class ResearchStatsEngine:
    """Rigorous statistical validation for Tier-1 research publishing."""
    
    @staticmethod
    def calculate_significance(baseline_scores: list, proposed_scores: list):
        """Calculates p-value and Cohen's d (Effect Size)."""
        # Paired T-test (assuming related samples under noise levels)
        t_stat, p_value = stats.ttest_rel(baseline_scores, proposed_scores)
        
        # Cohen's d (Effect Size)
        # d = (mean1 - mean2) / pooled_std
        mean_diff = np.mean(proposed_scores) - np.mean(baseline_scores)
        pooled_std = np.sqrt((np.std(baseline_scores)**2 + np.std(proposed_scores)**2) / 2)
        cohens_d = mean_diff / (pooled_std + 1e-9)
        
        return {
            "p_value": p_value,
            "t_stat": t_stat,
            "cohens_d": cohens_d,
            "is_significant": p_value < 0.05,
            "improvement_pct": (mean_diff / (np.mean(baseline_scores) + 1e-9)) * 100
        }

    def generate_research_table(self, baseline, proposed):
        stats_dict = self.calculate_significance(baseline, proposed)
        print("\n--- [Research Validation Table] ---")
        print(f"| Metric | Baseline | Proposed | Improvement | p-value | Cohen's d |")
        print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
        print(f"| Precision@1 | {np.mean(baseline):.3f} | {np.mean(proposed):.3f} | " +
              f"{stats_dict['improvement_pct']:.1f}% | {stats_dict['p_value']:.4f} | " +
              f"{stats_dict['cohens_d']:.2f} |")
        
        if stats_dict['p_value'] < 0.01:
            print("\n[Analysis]: The system achieves 'highly significant' (p < 0.01) " +
                  "robustness compared to standard RAG baselines.")
        elif stats_dict['p_value'] < 0.05:
            print("\n[Analysis]: The system achieves 'statistically significant' (p < 0.05) " +
                  "robustness compared to standard RAG baselines.")
        else:
            print("\n[Analysis]: Baseline vs. Proposed difference is not yet significant " +
                  "at the current sample size.")
