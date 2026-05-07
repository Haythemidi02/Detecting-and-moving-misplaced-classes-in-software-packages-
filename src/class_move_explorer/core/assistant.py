"""
MoveClassAssistant - Orchestrates the entire class placement analysis
"""
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

from class_move_explorer.analyzers.java_parser import JavaProjectAnalyzer
from class_move_explorer.analyzers.dependency_analyzer import DependencyAnalyzer
from class_move_explorer.analyzers.embedding_analyzer import EmbeddingAnalyzer
from class_move_explorer.analyzers.llm_analyzer import LLMAnalyzer
from class_move_explorer.utils.metrics import PerformanceMetrics


class MoveClassAssistant:
    def __init__(self):
        self.project_analyzer = JavaProjectAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.embedding_analyzer = EmbeddingAnalyzer()
        self.llm_analyzer = LLMAnalyzer()
        self.metrics = PerformanceMetrics()
        
    def analyze_and_recommend(self, project_path: str, output_csv: str = "class_placement_analysis.csv") -> pd.DataFrame:
        """Main entry point for analyzing Java project and generating recommendations"""
        print(f"\nStarting Analysis: {project_path}")
        start_time = datetime.now()
        
        # 1. Parse Project
        classes_data = self.project_analyzer.analyze_project(project_path)
        if not classes_data:
            print("No Java classes found.")
            return pd.DataFrame()
            
        # 2. Structural Analysis
        dependency_graph = (
            self.dependency_analyzer.analyze_dependencies(classes_data)
            if self.dependency_analyzer is not None
            else {}
        )
        
        # 3. Semantic Analysis
        embeddings = (
            self.embedding_analyzer.compute_embeddings(classes_data)
            if self.embedding_analyzer is not None
            else {}
        )
        
        # 4. Identification
        misplaced_classes = self.llm_analyzer.identify_misplaced_classes(
            classes_data, dependency_graph, embeddings
        )
        
        # 5. Suggestion Generation
        suggestions = self.llm_analyzer.suggest_target_packages(
            misplaced_classes, classes_data, dependency_graph, embeddings
        )
        
        # 6. Formatting
        results_df = self._format_results(classes_data, misplaced_classes, suggestions, start_time)
        
        # 7. Output
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            
        # 8. Metrics
        metrics_results = self.metrics.calculate_metrics(results_df)
        self.metrics.display_metrics(metrics_results)
        
        return results_df
    
    def _format_results(self, classes_data: List[Dict], 
                        misplaced: List[str], 
                        suggestions: Dict[str, Dict],
                        start_time: datetime) -> pd.DataFrame:
        """Format results for the report"""
        results = []
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        for class_info in classes_data:
            name = class_info['class_name']
            is_misplaced = name in misplaced
            
            row = {
                'class_name': name,
                'current_package': class_info['package'],
                'is_misplaced': is_misplaced,
                'suggested_package': suggestions[name]['suggested_package'] if is_misplaced else class_info['package'],
                'confidence': suggestions[name]['confidence'] if is_misplaced else 0.95,
                'reasoning': suggestions[name]['reasoning'] if is_misplaced else 'Correctly placed',
                'method_used': suggestions[name]['method_used'] if is_misplaced else 'n/a',
                'analysis_time_sec': round(analysis_time, 2)
            }
            results.append(row)
            
        return pd.DataFrame(results)
