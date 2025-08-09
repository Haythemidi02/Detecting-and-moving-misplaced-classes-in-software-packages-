"""
DependencyAnalyzer - Analyzes dependencies between Java classes
"""
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter


class DependencyAnalyzer:
    def __init__(self):
        self.dependency_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.package_dependencies = defaultdict(set)
    
    def analyze_dependencies(self, classes_data: List[Dict]) -> Dict:
        """Analyze dependencies between classes and packages"""
        # Build class name to package mapping
        class_to_package = {cls['class_name']: cls['package'] for cls in classes_data}
        
        # Analyze each class
        for class_info in classes_data:
            class_name = class_info['class_name']
            package = class_info['package']
            
            # Analyze imports
            for import_stmt in class_info['imports']:
                imported_class = import_stmt.split('.')[-1]
                if imported_class in class_to_package:
                    self.dependency_graph[class_name].add(imported_class)
                    self.reverse_graph[imported_class].add(class_name)
                    
                    # Package-level dependencies
                    imported_package = class_to_package[imported_class]
                    if package != imported_package:
                        self.package_dependencies[package].add(imported_package)
            
            # Analyze extends and implements
            if class_info['extends'] and class_info['extends'] in class_to_package:
                parent_class = class_info['extends']
                self.dependency_graph[class_name].add(parent_class)
                self.reverse_graph[parent_class].add(class_name)
            
            for interface in class_info['implements']:
                if interface in class_to_package:
                    self.dependency_graph[class_name].add(interface)
                    self.reverse_graph[interface].add(class_name)
        
        # Calculate additional metrics
        dependency_metrics = self._calculate_dependency_metrics(classes_data)
        
        return {
            'class_dependencies': dict(self.dependency_graph),
            'reverse_dependencies': dict(self.reverse_graph),
            'package_dependencies': dict(self.package_dependencies),
            'metrics': dependency_metrics
        }
    
    def _calculate_dependency_metrics(self, classes_data: List[Dict]) -> Dict:
        """Calculate dependency-related metrics"""
        package_cohesion = self._calculate_package_cohesion(classes_data)
        coupling_metrics = self._calculate_coupling_metrics(classes_data)
        
        return {
            'package_cohesion': package_cohesion,
            'coupling_metrics': coupling_metrics,
            'total_dependencies': sum(len(deps) for deps in self.dependency_graph.values()),
            'cyclic_dependencies': self._detect_cycles()
        }
    
    def _calculate_package_cohesion(self, classes_data: List[Dict]) -> Dict[str, float]:
        """Calculate cohesion score for each package"""
        package_classes = defaultdict(list)
        for cls in classes_data:
            package_classes[cls['package']].append(cls['class_name'])
        
        cohesion_scores = {}
        for package, classes in package_classes.items():
            if len(classes) <= 1:
                cohesion_scores[package] = 1.0
                continue
            
            # Calculate internal vs external dependencies
            internal_deps = 0
            external_deps = 0
            
            for class_name in classes:
                for dep in self.dependency_graph.get(class_name, set()):
                    if dep in classes:
                        internal_deps += 1
                    else:
                        external_deps += 1
            
            total_deps = internal_deps + external_deps
            cohesion_scores[package] = internal_deps / total_deps if total_deps > 0 else 1.0
        
        return cohesion_scores
    
    def _calculate_coupling_metrics(self, classes_data: List[Dict]) -> Dict:
        """Calculate coupling metrics"""
        class_to_package = {cls['class_name']: cls['package'] for cls in classes_data}
        
        afferent_coupling = defaultdict(int)  # Ca - incoming dependencies
        efferent_coupling = defaultdict(int)  # Ce - outgoing dependencies
        
        for package in set(cls['package'] for cls in classes_data):
            package_classes = [cls['class_name'] for cls in classes_data if cls['package'] == package]
            
            for class_name in package_classes:
                # Efferent coupling - dependencies going out of the package
                for dep in self.dependency_graph.get(class_name, set()):
                    if dep in class_to_package and class_to_package[dep] != package:
                        efferent_coupling[package] += 1
                
                # Afferent coupling - dependencies coming into the package
                for dep in self.reverse_graph.get(class_name, set()):
                    if dep in class_to_package and class_to_package[dep] != package:
                        afferent_coupling[package] += 1
        
        return {
            'afferent_coupling': dict(afferent_coupling),
            'efferent_coupling': dict(efferent_coupling)
        }
    
    def _detect_cycles(self) -> List[List[str]]:
        """Detect cyclic dependencies using DFS"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
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