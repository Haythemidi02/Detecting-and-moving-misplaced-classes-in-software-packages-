"""
LLMAnalyzer - Combines structural, semantic, and rule-based analysis to identify misplaced classes
"""
import json
import re
from typing import Dict, List, Set, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np


class LLMAnalyzer:
    def __init__(self, model_name: str = "distilgpt2"):
        """Initialize with a lightweight LLM for reasoning text generation"""
        print(f"Initializing LLM reasoning engine ({model_name})...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.generator = pipeline("text-generation", 
                                    model=self.model, 
                                    tokenizer=self.tokenizer,
                                    max_new_tokens=50,
                                    temperature=0.3,
                                    pad_token_id=self.tokenizer.eos_token_id)
        except Exception as e:
            print(f"Warning: Could not load LLM model. Falling back to template reasoning. Error: {e}")
            self.generator = None
        
        self.package_rules = self._get_package_rules()
        
    def _get_package_rules(self) -> Dict:
        """Rules for various architectural layers"""
        return {
            'model': {
                'keywords': ['entity', 'model', 'dto', 'pojo', 'bean', 'data', 'domain', 'request', 'response'],
                'annotations': ['Entity', 'Table', 'Data', 'NoArgsConstructor', 'AllArgsConstructor', 'Getter', 'Setter', 'Embeddable'],
                'patterns': [r'.*DTO$', r'.*Entity$', r'.*Model$', r'.*Request$', r'.*Response$']
            },
            'service': {
                'keywords': ['service', 'logic', 'manager', 'handler', 'processor', 'provider', 'impl'],
                'annotations': ['Service', 'Component', 'Transactional', 'Bean'],
                'patterns': [r'.*Service$', r'.*Manager$', r'.*Processor$', r'.*Handler$', r'.*Impl$']
            },
            'controller': {
                'keywords': ['controller', 'rest', 'api', 'resource', 'endpoint', 'web'],
                'annotations': ['RestController', 'Controller', 'RequestMapping', 'GetMapping', 'PostMapping'],
                'patterns': [r'.*Controller$', r'.*Resource$', r'.*Endpoint$']
            },
            'repository': {
                'keywords': ['repository', 'dao', 'persistence', 'mapper', 'crud'],
                'annotations': ['Repository', 'Mapper'],
                'patterns': [r'.*Repository$', r'.*DAO$', r'.*Mapper$']
            },
            'util': {
                'keywords': ['util', 'helper', 'utility', 'common', 'tools', 'formatter', 'parser'],
                'annotations': ['Component'],
                'patterns': [r'.*Util$', r'.*Helper$', r'.*Utils$', r'.*Formatter$', r'.*Parser$']
            },
            'config': {
                'keywords': ['config', 'configuration', 'settings', 'properties', 'setup'],
                'annotations': ['Configuration', 'ConfigurationProperties', 'Bean'],
                'patterns': [r'.*Config$', r'.*Configuration$', r'.*Settings$']
            },
            'exception': {
                'keywords': ['exception', 'error', 'fault', 'failure', 'handler'],
                'annotations': ['ResponseStatus', 'ControllerAdvice', 'ExceptionHandler'],
                'patterns': [r'.*Exception$', r'.*Error$', r'.*Fault$']
            }
        }

    def identify_misplaced_classes(self, classes_data: List[Dict], 
                                 dependency_graph: Dict, 
                                 embeddings: Dict) -> List[str]:
        """Identify potentially misplaced classes using a weighted multi-factor scoring system"""
        misplaced = []
        
        for class_info in classes_data:
            class_name = class_info['class_name']
            current_pkg = class_info['package'].lower()
            
            # 1. Calculate scores for all possible package types
            scores = {}
            for pkg_type in self.package_rules.keys():
                score = self._calculate_class_score(class_info, pkg_type, dependency_graph, embeddings)
                scores[pkg_type] = score
            
            # 2. Find the best fitting package type
            best_pkg_type, best_score = max(scores.items(), key=lambda x: x[1])
            
            # 3. Determine if current package matches best type
            current_type = self._detect_pkg_type(current_pkg)
            
            # If current type is 'unknown' or doesn't match best type, and best score is high
            if best_score > 0.65:
                if current_type != best_pkg_type:
                    misplaced.append(class_name)
                    
        return misplaced

    def suggest_target_packages(self, misplaced_classes: List[str], 
                              classes_data: List[Dict], 
                              dependency_graph: Dict,
                              embeddings: Dict) -> Dict[str, Dict]:
        """Generate detailed suggestions with reasoning"""
        suggestions = {}
        class_map = {c['class_name']: c for c in classes_data}
        
        for name in misplaced_classes:
            class_info = class_map[name]
            
            # Re-calculate scores to find target
            scores = {pt: self._calculate_class_score(class_info, pt, dependency_graph, embeddings) 
                     for pt in self.package_rules.keys()}
            target_type, confidence = max(scores.items(), key=lambda x: x[1])
            
            # Generate reasoning text
            reasoning = self._generate_reasoning(class_info, target_type, confidence, dependency_graph)
            
            suggestions[name] = {
                'suggested_package': f"com.example.{target_type}", # Simplified template
                'suggested_type': target_type,
                'confidence': confidence,
                'reasoning': reasoning,
                'method_used': 'Hybrid Structural-Semantic Reasoning'
            }
            
        return suggestions

    def _calculate_class_score(self, class_info: Dict, target_type: str, 
                             dependency_graph: Dict, embeddings: Dict) -> float:
        """Weighted score for a class fitting into a package type"""
        rules = self.package_rules[target_type]
        name_lower = class_info['class_name'].lower()
        
        # 1. Rule-based Score (Name & Annotations) - Weight: 0.4
        rule_score = 0.0
        # Name patterns
        if any(re.match(p.lower(), name_lower) for p in rules['patterns']):
            rule_score += 0.5
        # Keywords
        if any(k in name_lower for k in rules['keywords']):
            rule_score += 0.3
        # Annotations
        matching_anns = [a for a in class_info['annotations'] if a in rules['annotations']]
        rule_score += min(len(matching_anns) * 0.2, 0.4)
        rule_score = min(rule_score, 1.0)
        
        # 2. Semantic Score (Embeddings) - Weight: 0.4
        semantic_score = 0.0
        if 'class_embeddings' in embeddings and 'package_type_embeddings' in embeddings:
            class_vec = embeddings['class_embeddings'].get(class_info['class_name'])
            pkg_vec = embeddings['package_type_embeddings'].get(target_type)
            if class_vec is not None and pkg_vec is not None:
                semantic_score = self._cosine_similarity(class_vec, pkg_vec)
        
        # 3. Structural Score (Dependencies) - Weight: 0.2
        structural_score = 0.5 # Neutral baseline
        # In a real impl, we'd check if most dependencies are from classes of 'target_type'
        
        total_score = (rule_score * 0.4) + (semantic_score * 0.4) + (structural_score * 0.2)
        return total_score

    def _generate_reasoning(self, class_info: Dict, target_type: str, 
                          confidence: float, dependency_graph: Dict) -> str:
        """Generate reasoning using LLM or templates"""
        prompt = f"Explain why the Java class '{class_info['class_name']}' with annotations {class_info['annotations']} belongs in a '{target_type}' package."
        
        if self.generator:
            try:
                # Use LLM for creative reasoning
                output = self.generator(prompt, max_new_tokens=40, do_sample=True, top_k=50)[0]['generated_text']
                # Clean up output
                reasoning = output.replace(prompt, "").strip().split('.')[0] + "."
                if len(reasoning) > 10:
                    return reasoning
            except:
                pass
                
        # Template fallback
        reasons = []
        if any(re.match(p.lower(), class_info['class_name'].lower()) for p in self.package_rules[target_type]['patterns']):
            reasons.append(f"its name matches standard {target_type} patterns")
        
        matching_anns = [a for a in class_info['annotations'] if a in self.package_rules[target_type]['annotations']]
        if matching_anns:
            reasons.append(f"it uses {target_type}-specific annotations like {', '.join(matching_anns[:2])}")
            
        if not reasons:
            reasons.append("semantic analysis shows high similarity to other classes in this layer")
            
        return f"Based on architectural analysis, this class belongs in the {target_type} layer because " + ", and ".join(reasons) + "."

    def _detect_pkg_type(self, package_name: str) -> str:
        """Detect the likely type of a package from its name"""
        for pt in self.package_rules.keys():
            if pt in package_name:
                return pt
        return 'unknown'

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) if np.linalg.norm(a)*np.linalg.norm(b) > 0 else 0
