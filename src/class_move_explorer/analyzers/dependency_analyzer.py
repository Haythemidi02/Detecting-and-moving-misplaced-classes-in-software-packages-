"""
DependencyAnalyzer - Analyzes structural dependencies and computes cohesion/coupling metrics
"""
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter


class DependencyAnalyzer:
    def __init__(self):
        self.dependency_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.package_dependencies = defaultdict(set)
        self.class_to_package = {}
    
    def analyze_dependencies(self, classes_data: List[Dict]) -> Dict:
        """Analyze dependencies between classes and packages with improved accuracy"""
        self.class_to_package = {cls['class_name']: cls['package'] for cls in classes_data}
        
        # Analyze each class
        for class_info in classes_data:
            class_name = class_info['class_name']
            package = class_info['package']
            
            # 1. Analyze imports (direct dependencies)
            for import_stmt in class_info['imports']:
                imported_class = import_stmt.split('.')[-1]
                self._add_dependency(class_name, imported_class)
            
            # 2. Analyze inheritance
            if class_info['extends']:
                self._add_dependency(class_name, class_info['extends'])
            
            for interface in class_info['implements']:
                self._add_dependency(class_name, interface)
            
            # 3. Analyze field types
            for field in class_info.get('fields_detailed', []):
                self._add_dependency(class_name, field['type'])
            
            # 4. Analyze method parameters and returns
            for method in class_info.get('methods_detailed', []):
                self._add_dependency(class_name, method['return_type'])
                for param_type in method['parameters']:
                    self._add_dependency(class_name, param_type)
        
        # Calculate additional metrics
        dependency_metrics = self._calculate_dependency_metrics(classes_data)
        
        return {
            'class_dependencies': {k: list(v) for k, v in self.dependency_graph.items()},
            'reverse_dependencies': {k: list(v) for k, v in self.reverse_graph.items()},
            'package_dependencies': {k: list(v) for k, v in self.package_dependencies.items()},
            'metrics': dependency_metrics
        }
    
    def _add_dependency(self, from_class: str, to_class: str):
        """Helper to add a dependency if the target class is within our project"""
        if to_class in self.class_to_package and from_class != to_class:
            self.dependency_graph[from_class].add(to_class)
            self.reverse_graph[to_class].add(from_class)
            
            # Package-level
            from_pkg = self.class_to_package[from_class]
            to_pkg = self.class_to_package[to_class]
            if from_pkg != to_pkg:
                self.package_dependencies[from_pkg].add(to_pkg)

    def _calculate_dependency_metrics(self, classes_data: List[Dict]) -> Dict:
        """Calculate advanced metrics for cohesion and coupling"""
        package_cohesion = self._calculate_package_cohesion(classes_data)
        coupling_metrics = self._calculate_coupling_metrics(classes_data)
        class_metrics = self._calculate_class_metrics(classes_data)
        
        return {
            'package_cohesion': package_cohesion,
            'coupling_metrics': coupling_metrics,
            'class_metrics': class_metrics,
            'total_dependencies': sum(len(deps) for deps in self.dependency_graph.values()),
            'cyclic_dependencies': self._detect_cycles()
        }
    
    def _calculate_package_cohesion(self, classes_data: List[Dict]) -> Dict[str, float]:
        """
        Calculate cohesion for each package.
        Cohesion = Internal Dependencies / (Internal + External Dependencies)
        """
        package_classes = defaultdict(list)
        for cls in classes_data:
            package_classes[cls['package']].append(cls['class_name'])
        
        cohesion_scores = {}
        for package, classes in package_classes.items():
            if len(classes) <= 1:
                cohesion_scores[package] = 1.0
                continue
            
            internal_deps = 0
            external_deps = 0
            
            for class_name in classes:
                for dep in self.dependency_graph.get(class_name, set()):
                    if dep in classes:
                        internal_deps += 1
                    else:
                        external_deps += 1
            
            total_deps = internal_deps + external_deps
            cohesion_scores[package] = internal_deps / total_deps if total_deps > 0 else 0.5
        
        return cohesion_scores
    
    def _calculate_coupling_metrics(self, classes_data: List[Dict]) -> Dict:
        """Calculate Afferent (Ca) and Efferent (Ce) coupling for packages"""
        ca = defaultdict(set)  # Classes outside package that depend on classes inside
        ce = defaultdict(set)  # Classes inside package that depend on classes outside
        
        for class_name, deps in self.dependency_graph.items():
            pkg_from = self.class_to_package.get(class_name)
            for dep in deps:
                pkg_to = self.class_to_package.get(dep)
                if pkg_from and pkg_to and pkg_from != pkg_to:
                    ce[pkg_from].add(pkg_to)
                    ca[pkg_to].add(pkg_from)
        
        return {
            'afferent_coupling': {pkg: len(pkgs) for pkg, pkgs in ca.items()},
            'efferent_coupling': {pkg: len(pkgs) for pkg, pkgs in ce.items()}
        }

    def _calculate_class_metrics(self, classes_data: List[Dict]) -> Dict:
        """Calculate class-level metrics"""
        metrics = {}
        for class_info in classes_data:
            name = class_info['class_name']
            out_deps = len(self.dependency_graph.get(name, []))
            in_deps = len(self.reverse_graph.get(name, []))
            
            metrics[name] = {
                'out_degree': out_deps,
                'in_degree': in_deps,
                'total_degree': out_deps + in_deps
            }
        return metrics

    def _detect_cycles(self) -> List[List[str]]:
        """Detect cyclic dependencies using DFS"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for node in self.dependency_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
