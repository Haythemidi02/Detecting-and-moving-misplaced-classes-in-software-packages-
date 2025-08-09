"""
PerformanceMetrics - Calculates and displays performance metrics for the analysis
"""
import pandas as pd
from typing import Dict, List
from collections import Counter
import numpy as np


class PerformanceMetrics:
    def __init__(self):
        self.metrics_calculated = {}
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        total_classes = len(results_df)
        misplaced_classes = len(results_df[results_df['is_misplaced'] == True])
        correctly_placed = total_classes - misplaced_classes
        
        # Basic metrics
        metrics = {
            'total_classes_analyzed': total_classes,
            'misplaced_classes_found': misplaced_classes,
            'correctly_placed_classes': correctly_placed,
            'misplacement_rate': misplaced_classes / total_classes if total_classes > 0 else 0,
            'accuracy_rate': correctly_placed / total_classes if total_classes > 0 else 0
        }
        
        # Confidence metrics
        if misplaced_classes > 0:
            misplaced_df = results_df[results_df['is_misplaced'] == True]
            metrics.update({
                'average_confidence': misplaced_df['confidence'].mean(),
                'min_confidence': misplaced_df['confidence'].min(),
                'max_confidence': misplaced_df['confidence'].max(),
                'confidence_std': misplaced_df['confidence'].std()
            })
        
        # Package distribution metrics
        package_metrics = self._calculate_package_metrics(results_df)
        metrics.update(package_metrics)
        
        # Quality metrics
        quality_metrics = self._calculate_quality_metrics(results_df)
        metrics.update(quality_metrics)
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(results_df)
        metrics.update(performance_metrics)
        
        self.metrics_calculated = metrics
        return metrics
    
    def _calculate_package_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate package-related metrics"""
        # Current package distribution
        current_packages = Counter(results_df['current_package'])
        
        # Suggested package distribution for misplaced classes
        misplaced_df = results_df[results_df['is_misplaced'] == True]
        suggested_packages = Counter(misplaced_df['suggested_package']) if len(misplaced_df) > 0 else {}
        
        # Package misplacement rates
        package_misplacement = {}
        for package in current_packages:
            package_classes = results_df[results_df['current_package'] == package]
            misplaced_in_package = len(package_classes[package_classes['is_misplaced'] == True])
            package_misplacement[package] = {
                'total_classes': len(package_classes),
                'misplaced_classes': misplaced_in_package,
                'misplacement_rate': misplaced_in_package / len(package_classes)
            }
        
        return {
            'package_distribution': dict(current_packages),
            'suggested_package_distribution': dict(suggested_packages),
            'package_misplacement_rates': package_misplacement,
            'most_problematic_package': max(package_misplacement.items(), 
                                          key=lambda x: x[1]['misplacement_rate'])[0] if package_misplacement else None
        }
    
    def _calculate_quality_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate quality-related metrics"""
        misplaced_df = results_df[results_df['is_misplaced'] == True]
        
        if len(misplaced_df) == 0:
            return {
                'high_confidence_suggestions': 0,
                'medium_confidence_suggestions': 0,
                'low_confidence_suggestions': 0,
                'reasoning_quality_score': 1.0
            }
        
        # Confidence distribution
        high_conf = len(misplaced_df[misplaced_df['confidence'] >= 0.8])
        medium_conf = len(misplaced_df[(misplaced_df['confidence'] >= 0.6) & (misplaced_df['confidence'] < 0.8)])
        low_conf = len(misplaced_df[misplaced_df['confidence'] < 0.6])
        
        # Reasoning quality (based on length and content)
        reasoning_scores = []
        for reasoning in misplaced_df['reasoning']:
            score = min(len(reasoning.split()) / 10, 1.0)  # Normalize by word count
            if any(keyword in reasoning.lower() for keyword in ['annotation', 'method', 'pattern', 'naming']):
                score += 0.2
            reasoning_scores.append(min(score, 1.0))
        
        return {
            'high_confidence_suggestions': high_conf,
            'medium_confidence_suggestions': medium_conf,
            'low_confidence_suggestions': low_conf,
            'reasoning_quality_score': np.mean(reasoning_scores) if reasoning_scores else 0.0
        }
    
    def _calculate_performance_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate performance-related metrics"""
        if len(results_df) == 0:
            return {}
        
        # Analysis time metrics
        analysis_times = results_df['analysis_time'].dropna()
        
        metrics = {}
        if len(analysis_times) > 0:
            metrics.update({
                'average_analysis_time': analysis_times.mean(),
                'total_analysis_time': analysis_times.iloc[0],  # Same for all rows
                'analysis_time_per_class': analysis_times.mean() / len(results_df) if len(results_df) > 0 else 0
            })
        
        # Method distribution
        methods_used = Counter(results_df['method_used'])
        metrics['analysis_methods_used'] = dict(methods_used)
        
        return metrics
    
    def display_metrics(self, metrics: Dict):
        """Display metrics in a formatted way"""
        print("\n" + "="*60)
        print("CLASS PLACEMENT ANALYSIS METRICS")
        print("="*60)
        
        # Basic metrics
        print(f"\n📊 OVERVIEW:")
        print(f"   Total classes analyzed: {metrics.get('total_classes_analyzed', 0)}")
        print(f"   Misplaced classes found: {metrics.get('misplaced_classes_found', 0)}")
        print(f"   Correctly placed classes: {metrics.get('correctly_placed_classes', 0)}")
        print(f"   Misplacement rate: {metrics.get('misplacement_rate', 0):.2%}")
        print(f"   Overall accuracy: {metrics.get('accuracy_rate', 0):.2%}")
        
        # Confidence metrics
        if 'average_confidence' in metrics:
            print(f"\n🎯 CONFIDENCE METRICS:")
            print(f"   Average confidence: {metrics['average_confidence']:.3f}")
            print(f"   Confidence range: {metrics.get('min_confidence', 0):.3f} - {metrics.get('max_confidence', 0):.3f}")
            print(f"   Confidence std dev: {metrics.get('confidence_std', 0):.3f}")
        
        # Quality metrics
        print(f"\n⭐ QUALITY METRICS:")
        print(f"   High confidence suggestions (≥0.8): {metrics.get('high_confidence_suggestions', 0)}")
        print(f"   Medium confidence suggestions (0.6-0.8): {metrics.get('medium_confidence_suggestions', 0)}")
        print(f"   Low confidence suggestions (<0.6): {metrics.get('low_confidence_suggestions', 0)}")
        print(f"   Reasoning quality score: {metrics.get('reasoning_quality_score', 0):.3f}")
        
        # Package metrics
        if 'most_problematic_package' in metrics and metrics['most_problematic_package']:
            print(f"\n📦 PACKAGE ANALYSIS:")
            print(f"   Most problematic package: {metrics['most_problematic_package']}")
            
            package_dist = metrics.get('package_distribution', {})
            if package_dist:
                print(f"   Package distribution:")
                for package, count in sorted(package_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     {package}: {count} classes")
        
        # Performance metrics
        if 'total_analysis_time' in metrics:
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"   Total analysis time: {metrics['total_analysis_time']:.2f} seconds")
            print(f"   Average time per class: {metrics.get('analysis_time_per_class', 0):.4f} seconds")
        
        print("\n" + "="*60)
    
    def export_detailed_metrics(self, output_file: str = "analysis_metrics.json"):
        """Export detailed metrics to JSON file"""
        import json
        
        if self.metrics_calculated:
            # Convert numpy types to native Python types for JSON serialization
            serializable_metrics = self._make_serializable(self.metrics_calculated)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            
            print(f"Detailed metrics exported to {output_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report"""
        if not self.metrics_calculated:
            return "No metrics calculated yet."
        
        metrics = self.metrics_calculated
        
        report = f"""
CLASS PLACEMENT ANALYSIS SUMMARY
================================

OVERVIEW:
- Analyzed {metrics.get('total_classes_analyzed', 0)} classes
- Found {metrics.get('misplaced_classes_found', 0)} potentially misplaced classes
- Overall misplacement rate: {metrics.get('misplacement_rate', 0):.1%}
- Analysis accuracy: {metrics.get('accuracy_rate', 0):.1%}

CONFIDENCE ANALYSIS:
- High confidence suggestions: {metrics.get('high_confidence_suggestions', 0)}
- Medium confidence suggestions: {metrics.get('medium_confidence_suggestions', 0)}  
- Low confidence suggestions: {metrics.get('low_confidence_suggestions', 0)}
- Average confidence: {metrics.get('average_confidence', 0):.3f}

RECOMMENDATIONS:
- Focus on high confidence suggestions first
- Review package structure for most problematic packages
- Consider refactoring based on semantic clustering results

"""
        return report