"""
PerformanceMetrics - Calculates and displays comprehensive metrics for analysis and evaluation
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from collections import Counter


class PerformanceMetrics:
    def __init__(self):
        self.metrics_calculated = {}
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate analysis metrics based on tool results"""
        if results_df.empty:
            return {}
            
        total = len(results_df)
        misplaced = results_df[results_df['is_misplaced'] == True]
        
        metrics = {
            'total_classes': total,
            'misplaced_count': len(misplaced),
            'misplacement_rate': len(misplaced) / total,
            'average_confidence': misplaced['confidence'].mean() if not misplaced.empty else 0.0,
            'confidence_by_method': results_df.groupby('method_used')['confidence'].mean().to_dict()
        }
        
        # Reasoning quality score (heuristic)
        if not misplaced.empty:
            metrics['reasoning_completeness'] = misplaced['reasoning'].apply(lambda x: len(x.split()) / 20).mean()
            
        self.metrics_calculated = metrics
        return metrics
    
    def display_metrics(self, metrics: Dict):
        """Display formatted metrics to the console"""
        print("\n" + "="*50)
        print(" CLASS PLACEMENT ANALYSIS METRICS ".center(50, "="))
        print("="*50)
        
        if not metrics:
            print("No data available.")
            return

        print(f"[*] OVERVIEW:")
        print(f"   - Total Classes:      {metrics.get('total_classes', 0)}")
        print(f"   - Misplaced Found:    {metrics.get('misplaced_count', 0)}")
        print(f"   - Misplacement Rate:  {metrics.get('misplacement_rate', 0):.1%}")
        
        if 'average_confidence' in metrics:
            print(f"\n[*] QUALITY:")
            print(f"   - Avg Confidence:     {metrics['average_confidence']:.2f}")
            if 'reasoning_completeness' in metrics:
                print(f"   - Reasoning Quality:  {min(metrics['reasoning_completeness'], 1.0):.2f}")

        if 'precision' in metrics:
            print(f"\n[*] EVALUATION (vs Ground Truth):")
            print(f"   - Precision:          {metrics['precision']:.2%}")
            print(f"   - Recall:             {metrics['recall']:.2%}")
            print(f"   - F1 Score:           {metrics['f1_score']:.2%}")
            print(f"   - Suggestion Acc:     {metrics['suggestion_accuracy']:.2%}")

        print("="*50 + "\n")

    def display_evaluation_summary(self, eval_results: Dict):
        """Specialized display for evaluation mode"""
        self.display_metrics(eval_results)
