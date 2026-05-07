"""
Evaluator - Formal evaluation framework for ClassMoveExplorer
"""
import random
import pandas as pd
from typing import List, Dict, Tuple
from class_move_explorer.core.assistant import MoveClassAssistant
from class_move_explorer.evaluation.synthetic_benchmark import (
    SyntheticProjectSpec,
    generate_synthetic_project,
    corrupt_packages,
)


class Evaluator:
    def __init__(self, assistant: MoveClassAssistant):
        self.assistant = assistant
        self.ground_truth = {} # class_name -> original_package
        
    def run_evaluation(self, project_path: str, misplace_ratio: float = 0.15) -> Dict:
        """
        Run evaluation by intentionally misplacing classes and measuring recovery performance
        """
        print(f"Starting evaluation on: {project_path}")
        # Reset per-run ground truth (avoid leakage across repeated evaluations)
        self.ground_truth = {}
        
        # 1. Analyze project to get current (correct) state
        classes_data = self.assistant.project_analyzer.analyze_project(project_path)
        if not classes_data:
            return {"error": "No classes found to evaluate"}
            
        # 2. Select classes to misplace
        num_to_misplace = max(1, int(len(classes_data) * misplace_ratio))
        to_misplace_indices = random.sample(range(len(classes_data)), num_to_misplace)
        
        corrupted_data = []
        target_pkgs = ['com.example.controller', 'com.example.service', 'com.example.model', 'com.example.repository']
        
        print(f"Intentionally misplacing {num_to_misplace} classes...")
        
        for i, class_info in enumerate(classes_data):
            class_info_copy = class_info.copy()
            if i in to_misplace_indices:
                original_pkg = class_info['package']
                # Pick a random package that is different from original
                wrong_pkg = random.choice([p for p in target_pkgs if p not in original_pkg])
                
                self.ground_truth[class_info['class_name']] = original_pkg
                class_info_copy['package'] = wrong_pkg
                
            corrupted_data.append(class_info_copy)
            
        # 3. Run analysis on corrupted data
        # We need to bypass the file reading step and use our corrupted data directly
        # So we'll manually run the pipeline steps
        print("Running analysis on corrupted state...")
        dep_graph = (
            self.assistant.dependency_analyzer.analyze_dependencies(corrupted_data)
            if self.assistant.dependency_analyzer is not None
            else {}
        )
        embeddings = (
            self.assistant.embedding_analyzer.compute_embeddings(corrupted_data)
            if self.assistant.embedding_analyzer is not None
            else {}
        )
        
        misplaced_detected = self.assistant.llm_analyzer.identify_misplaced_classes(
            corrupted_data, dep_graph, embeddings
        )
        
        suggestions = self.assistant.llm_analyzer.suggest_target_packages(
            misplaced_detected, corrupted_data, dep_graph, embeddings
        )
        
        # 4. Calculate metrics
        metrics = self._calculate_performance_metrics(len(corrupted_data), misplaced_detected, suggestions)
        return metrics

    def _calculate_performance_metrics(self, total_count: int, detected: List[str], suggestions: Dict[str, Dict]) -> Dict:
        """Calculate Precision, Recall, and F1 based on ground truth"""
        ground_truth_names = set(self.ground_truth.keys())
        detected_names = set(detected)
        
        tp = len(ground_truth_names.intersection(detected_names))
        fp = len(detected_names - ground_truth_names)
        fn = len(ground_truth_names - detected_names)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Check suggestion accuracy
        correct_suggestions = 0
        for name in ground_truth_names:
            if name in suggestions:
                original_pkg = self.ground_truth[name]
                suggested_type = suggestions[name].get('suggested_type', '')
                if suggested_type in original_pkg.lower():
                    correct_suggestions += 1
                    
        suggestion_accuracy = correct_suggestions / tp if tp > 0 else 0
        
        return {
            "total_classes": total_count,
            "misplaced_count": len(detected),
            "misplaced_ground_truth": len(ground_truth_names),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "suggestion_accuracy": suggestion_accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "ground_truth_count": len(ground_truth_names)
        }

    def run_synthetic_benchmark(
        self,
        *,
        num_projects: int = 25,
        classes_per_type: int = 3,
        misplace_ratio: float = 0.25,
        seed: int = 42,
    ) -> Dict:
        """
        Repeatable benchmark that does NOT depend on user input:
        generate many synthetic well-placed projects, corrupt them, and measure recovery.
        """
        rng = random.Random(seed)
        spec = SyntheticProjectSpec(classes_per_type=classes_per_type, base_package="com.example")

        per_project: List[Dict] = []

        for _ in range(num_projects):
            project_seed = rng.randint(0, 10_000_000)
            classes = generate_synthetic_project(spec, seed=project_seed)
            corrupted, gt = corrupt_packages(classes, misplace_ratio=misplace_ratio, seed=project_seed + 1, base_package=spec.base_package)

            # Per-run ground truth
            self.ground_truth = gt

            dep_graph = (
                self.assistant.dependency_analyzer.analyze_dependencies(corrupted)
                if self.assistant.dependency_analyzer is not None
                else {}
            )
            embeddings = (
                self.assistant.embedding_analyzer.compute_embeddings(corrupted)
                if self.assistant.embedding_analyzer is not None
                else {}
            )

            detected = self.assistant.llm_analyzer.identify_misplaced_classes(corrupted, dep_graph, embeddings)
            suggestions = self.assistant.llm_analyzer.suggest_target_packages(detected, corrupted, dep_graph, embeddings)

            metrics = self._calculate_performance_metrics(len(corrupted), detected, suggestions)
            per_project.append(metrics)

        df = pd.DataFrame(per_project)
        if df.empty:
            return {"error": "Benchmark produced no results"}

        def _mean(col: str) -> float:
            return float(df[col].mean()) if col in df else 0.0

        def _std(col: str) -> float:
            return float(df[col].std(ddof=0)) if col in df else 0.0

        return {
            "mode": "synthetic_benchmark",
            "num_projects": num_projects,
            "classes_per_type": classes_per_type,
            "misplace_ratio": misplace_ratio,
            "seed": seed,
            "precision_mean": _mean("precision"),
            "recall_mean": _mean("recall"),
            "f1_mean": _mean("f1_score"),
            "suggestion_accuracy_mean": _mean("suggestion_accuracy"),
            "precision_std": _std("precision"),
            "recall_std": _std("recall"),
            "f1_std": _std("f1_score"),
            "suggestion_accuracy_std": _std("suggestion_accuracy"),
            "avg_total_classes": _mean("total_classes"),
            "avg_misplaced_ground_truth": _mean("misplaced_ground_truth"),
        }
