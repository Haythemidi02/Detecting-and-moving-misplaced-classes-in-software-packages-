"""
Main MoveClassAssistant - Orchestrates the entire class placement analysis
"""
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from java_project_analyzer import JavaProjectAnalyzer
from dependency_analyzer import DependencyAnalyzer
from embedding_analyzer import EmbeddingAnalyzer
from llm_analyzer import LLMAnalyzer
from metrics import PerformanceMetrics


class MoveClassAssistant:
    def __init__(self):
        self.project_analyzer = JavaProjectAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.embedding_analyzer = EmbeddingAnalyzer()
        self.llm_analyzer = LLMAnalyzer()
        self.metrics = PerformanceMetrics()
        
    def analyze_and_recommend(self, project_path: str, output_csv: str = "class_placement_analysis.csv") -> pd.DataFrame:
        """Main entry point for analyzing Java project and generating recommendations"""
        print(f"Starting analysis of project: {project_path}")
        start_time = datetime.now()
        
        # Step 1: Analyze project structure
        classes_data = self.project_analyzer.analyze_project(project_path)
        print(f"Found {len(classes_data)} classes")
        
        # Step 2: Build dependency graph
        dependency_graph = self.dependency_analyzer.analyze_dependencies(classes_data)
        print("Dependency analysis complete")
        
        # Step 3: Compute embeddings for semantic analysis
        embeddings = self.embedding_analyzer.compute_embeddings(classes_data)
        print("Embedding computation complete")
        
        # Step 4: Identify misplaced classes using LLM
        misplaced_classes = self.llm_analyzer.identify_misplaced_classes(
            classes_data, dependency_graph, embeddings
        )
        print(f"Identified {len(misplaced_classes)} potentially misplaced classes")
        
        # Step 5: Generate package suggestions
        suggestions = self.llm_analyzer.suggest_target_packages(
            misplaced_classes, classes_data, dependency_graph
        )
        
        # Step 6: Validate suggestions
        validated_results = self._validate_and_format_results(
            classes_data, misplaced_classes, suggestions, start_time
        )
        
        # Step 7: Save to CSV and calculate metrics
        df = pd.DataFrame(validated_results)
        df.to_csv(output_csv, index=False)
        
        # Calculate and display metrics
        metrics_results = self.metrics.calculate_metrics(df)
        self.metrics.display_metrics(metrics_results)
        
        print(f"Analysis complete. Results saved to {output_csv}")
        return df
    
    def _validate_and_format_results(self, classes_data: List[Dict], 
                                   misplaced_classes: List[str], 
                                   suggestions: Dict[str, Dict],
                                   start_time: datetime) -> List[Dict]:
        """Format results for CSV output"""
        results = []
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        for class_info in classes_data:
            class_id = class_info['class_name']
            is_misplaced = class_id in misplaced_classes
            
            if is_misplaced and class_id in suggestions:
                suggestion = suggestions[class_id]
                result = {
                    'class_id': class_id,
                    'class_name': class_info['class_name'],
                    'current_package': class_info['package'],
                    'is_misplaced': True,
                    'confidence': suggestion.get('confidence', 0.0),
                    'reasoning': suggestion.get('reasoning', ''),
                    'suggested_package': suggestion.get('suggested_package', ''),
                    'evidence': suggestion.get('evidence', ''),
                    'method_used': suggestion.get('method_used', 'llm+embedding'),
                    'analysis_time': round(analysis_time, 2),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            else:
                result = {
                    'class_id': class_id,
                    'class_name': class_info['class_name'],
                    'current_package': class_info['package'],
                    'is_misplaced': False,
                    'confidence': 0.95,
                    'reasoning': 'Class appears to be correctly placed',
                    'suggested_package': class_info['package'],
                    'evidence': 'No misplacement indicators found',
                    'method_used': 'llm+embedding',
                    'analysis_time': round(analysis_time, 2),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            
            results.append(result)
        
        return results


if __name__ == "__main__":
    assistant = MoveClassAssistant()
    # Example usage
    # assistant.analyze_and_recommend("/path/to/your/java/project")