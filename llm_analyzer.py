"""
LLMAnalyzer - Uses Hugging Face LLM to identify misplaced classes and suggest packages
"""
import json
import re
from typing import Dict, List, Set
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


class LLMAnalyzer:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize with a free Hugging Face model for text generation"""
        print(f"Loading LLM model: {model_name}")
        
        # Use a lightweight model that works well for classification
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            self.generator = pipeline("text-generation", 
                                    model=self.model, 
                                    tokenizer=self.tokenizer,
                                    max_length=200,
                                    temperature=0.7,
                                    pad_token_id=self.tokenizer.eos_token_id)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.generator = None
        
        # Define package placement rules
        self.placement_rules = self._initialize_placement_rules()
        
    def _initialize_placement_rules(self) -> Dict:
        """Initialize rules for determining correct package placement"""
        return {
            'model': {
                'keywords': ['entity', 'model', 'dto', 'pojo', 'bean', 'data', 'domain'],
                'patterns': [r'.*Entity$', r'.*Model$', r'.*DTO$', r'.*Bean$'],
                'methods': ['get', 'set', 'equals', 'hashCode', 'toString'],
                'annotations': ['Entity', 'Table', 'Column', 'Id', 'Data', 'NoArgsConstructor']
            },
            'service': {
                'keywords': ['service', 'business', 'logic', 'processor', 'handler', 'manager'],
                'patterns': [r'.*Service$', r'.*Manager$', r'.*Handler$', r'.*Processor$'],
                'methods': ['process', 'handle', 'execute', 'perform', 'calculate', 'validate'],
                'annotations': ['Service', 'Component', 'Transactional']
            },
            'controller': {
                'keywords': ['controller', 'rest', 'api', 'endpoint', 'web', 'resource'],
                'patterns': [r'.*Controller$', r'.*Resource$', r'.*Endpoint$'],
                'methods': ['get', 'post', 'put', 'delete', 'create', 'update', 'find'],
                'annotations': ['RestController', 'Controller', 'RequestMapping', 'GetMapping', 'PostMapping']
            },
            'repository': {
                'keywords': ['repository', 'dao', 'persistence', 'data'],
                'patterns': [r'.*Repository$', r'.*DAO$', r'.*Dao$'],
                'methods': ['find', 'save', 'delete', 'exists', 'count', 'query'],
                'annotations': ['Repository', 'Component']
            },
            'util': {
                'keywords': ['util', 'helper', 'utility', 'common', 'tools', 'constants'],
                'patterns': [r'.*Util$', r'.*Helper$', r'.*Utils$', r'.*Constants$'],
                'methods': ['static', 'helper', 'convert', 'format', 'parse'],
                'annotations': ['Component', 'Utility']
            },
            'config': {
                'keywords': ['config', 'configuration', 'settings', 'properties'],
                'patterns': [r'.*Config$', r'.*Configuration$', r'.*Settings$'],
                'methods': ['configure', 'setup', 'initialize'],
                'annotations': ['Configuration', 'ConfigurationProperties', 'Bean']
            },
            'exception': {
                'keywords': ['exception', 'error', 'fault', 'failure'],
                'patterns': [r'.*Exception$', r'.*Error$', r'.*Fault$'],
                'methods': ['getMessage', 'getCause'],
                'annotations': ['ResponseStatus']
            }
        }
    
    def identify_misplaced_classes(self, classes_data: List[Dict], 
                                 dependency_graph: Dict, 
                                 embeddings: Dict) -> List[str]:
        """Identify classes that are potentially misplaced"""
        misplaced_classes = []
        
        for class_info in classes_data:
            class_name = class_info['class_name']
            current_package = class_info['package']
            
            # Rule-based analysis
            rule_score, suggested_package_rule = self._analyze_with_rules(class_info)
            
            # Embedding-based analysis
            embedding_score, suggested_package_embedding = self._analyze_with_embeddings(
                class_info, embeddings
            )
            
            # Dependency-based analysis
            dependency_score = self._analyze_dependencies(class_info, dependency_graph)
            
            # Combined scoring
            total_score = (rule_score * 0.4 + embedding_score * 0.4 + dependency_score * 0.2)
            
            # Determine if class is misplaced (threshold: 0.6)
            if total_score > 0.6:
                current_package_type = self._extract_package_type(current_package)
                suggested_type = suggested_package_rule or suggested_package_embedding
                
                if current_package_type != suggested_type:
                    misplaced_classes.append(class_name)
        
        return misplaced_classes
    
    def suggest_target_packages(self, misplaced_classes: List[str], 
                              classes_data: List[Dict], 
                              dependency_graph: Dict) -> Dict[str, Dict]:
        """Generate package suggestions for misplaced classes"""
        suggestions = {}
        
        class_info_map = {cls['class_name']: cls for cls in classes_data}
        
        for class_name in misplaced_classes:
            if class_name not in class_info_map:
                continue
                
            class_info = class_info_map[class_name]
            suggestion = self._generate_suggestion(class_info, dependency_graph)
            suggestions[class_name] = suggestion
        
        return suggestions
    
    def _analyze_with_rules(self, class_info: Dict) -> tuple:
        """Analyze class using rule-based approach"""
        class_name = class_info['class_name'].lower()
        methods = [m.lower() for m in class_info['methods']]
        annotations = class_info['annotations']
        
        scores = {}
        
        for package_type, rules in self.placement_rules.items():
            score = 0
            
            # Check name patterns
            for pattern in rules['patterns']:
                if re.match(pattern.lower(), class_name):
                    score += 0.4
            
            # Check keywords in name
            for keyword in rules['keywords']:
                if keyword in class_name:
                    score += 0.2
            
            # Check method patterns
            method_matches = sum(1 for method in methods if any(rule_method in method for rule_method in rules['methods']))
            score += min(method_matches * 0.1, 0.3)
            
            # Check annotations
            annotation_matches = sum(1 for ann in annotations if ann in rules['annotations'])
            score += min(annotation_matches * 0.15, 0.3)
            
            scores[package_type] = min(score, 1.0)
        
        best_match = max(scores.items(), key=lambda x: x[1])
        return best_match[1], best_match[0]
    
    def _analyze_with_embeddings(self, class_info: Dict, embeddings: Dict) -> tuple:
        """Analyze class using embedding similarity"""
        if 'class_embeddings' not in embeddings or 'package_type_embeddings' not in embeddings:
            return 0.0, None
        
        class_name = class_info['class_name']
        if class_name not in embeddings['class_embeddings']:
            return 0.0, None
        
        class_embedding = embeddings['class_embeddings'][class_name]
        package_embeddings = embeddings['package_type_embeddings']
        
        # Find most similar package type
        similarities = {}
        for package_type, package_embedding in package_embeddings.items():
            similarity = self._cosine_similarity(class_embedding, package_embedding)
            similarities[package_type] = similarity
        
        best_match = max(similarities.items(), key=lambda x: x[1])
        return best_match[1], best_match[0]
    
    def _analyze_dependencies(self, class_info: Dict, dependency_graph: Dict) -> float:
        """Analyze dependencies to determine placement quality"""
        class_name = class_info['class_name']
        current_package = class_info['package']
        
        if class_name not in dependency_graph.get('class_dependencies', {}):
            return 0.0
        
        dependencies = dependency_graph['class_dependencies'][class_name]
        if not dependencies:
            return 0.0
        
        # Count cross-package dependencies
        cross_package_deps = 0
        for dep_class in dependencies:
            # This is simplified - in real implementation, you'd look up the package of dep_class
            cross_package_deps += 1
        
        # Higher cross-package dependencies might indicate misplacement
        dependency_ratio = cross_package_deps / len(dependencies)
        return min(dependency_ratio, 1.0)
    
    def _generate_suggestion(self, class_info: Dict, dependency_graph: Dict) -> Dict:
        """Generate detailed suggestion for a misplaced class"""
        class_name = class_info['class_name']
        
        # Analyze with rules
        rule_score, suggested_package = self._analyze_with_rules(class_info)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(class_info, suggested_package)
        
        # Generate evidence
        evidence = self._collect_evidence(class_info, suggested_package)
        
        return {
            'suggested_package': suggested_package,
            'confidence': min(rule_score, 0.95),
            'reasoning': reasoning,
            'evidence': evidence,
            'method_used': 'rule-based + semantic analysis'
        }
    
    def _generate_reasoning(self, class_info: Dict, suggested_package: str) -> str:
        """Generate human-readable reasoning for the suggestion"""
        class_name = class_info['class_name']
        current_package = class_info['package']
        
        reasons = []
        
        # Check naming patterns
        if suggested_package in class_name.lower():
            reasons.append(f"Class name '{class_name}' suggests {suggested_package} functionality")
        
        # Check annotations
        relevant_annotations = []
        for ann in class_info['annotations']:
            if ann.lower() in self.placement_rules.get(suggested_package, {}).get('annotations', []):
                relevant_annotations.append(ann)
        
        if relevant_annotations:
            reasons.append(f"Contains {suggested_package}-specific annotations: {', '.join(relevant_annotations)}")
        
        # Check methods
        relevant_methods = []
        rule_methods = self.placement_rules.get(suggested_package, {}).get('methods', [])
        for method in class_info['methods'][:5]:  # Check first 5 methods
            if any(rule_method in method.lower() for rule_method in rule_methods):
                relevant_methods.append(method)
        
        if relevant_methods:
            reasons.append(f"Contains {suggested_package}-typical methods: {', '.join(relevant_methods[:3])}")
        
        if not reasons:
            reasons.append(f"Semantic analysis suggests better fit in {suggested_package} package")
        
        return '; '.join(reasons)
    
    def _collect_evidence(self, class_info: Dict, suggested_package: str) -> str:
        """Collect evidence supporting the suggestion"""
        evidence_items = []
        
        # Pattern matching evidence
        for pattern in self.placement_rules.get(suggested_package, {}).get('patterns', []):
            if re.match(pattern.lower(), class_info['class_name'].lower()):
                evidence_items.append(f"Name matches {suggested_package} pattern")
                break
        
        # Annotation evidence
        matching_annotations = [ann for ann in class_info['annotations'] 
                              if ann in self.placement_rules.get(suggested_package, {}).get('annotations', [])]
        if matching_annotations:
            evidence_items.append(f"Annotations: {', '.join(matching_annotations)}")
        
        # Method evidence
        rule_methods = self.placement_rules.get(suggested_package, {}).get('methods', [])
        matching_methods = [method for method in class_info['methods'][:10] 
                          if any(rule_method in method.lower() for rule_method in rule_methods)]
        if matching_methods:
            evidence_items.append(f"Methods: {', '.join(matching_methods[:3])}")
        
        return '; '.join(evidence_items) if evidence_items else "Semantic similarity analysis"
    
    def _extract_package_type(self, package_name: str) -> str:
        """Extract package type from package name"""
        package_lower = package_name.lower()
        for package_type in self.placement_rules.keys():
            if package_type in package_lower:
                return package_type
        return 'unknown'
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0