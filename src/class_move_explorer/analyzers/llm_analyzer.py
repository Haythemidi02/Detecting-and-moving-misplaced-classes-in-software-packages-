"""
LLMAnalyzer - Combines structural, semantic, and rule-based analysis to identify misplaced classes
"""
import json
import re
import os
from typing import Dict, List, Set, Tuple
import numpy as np


DEFAULT_HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_HF_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]


class LLMAnalyzer:
    def __init__(self, use_huggingface: bool = True, hf_model: str = DEFAULT_HF_MODEL,
                 hf_api_token: str = None):
        """Initialize analyzer with HuggingFace API support"""
        self.hf_model = hf_model
        self.hf_api_token = hf_api_token
        self.hf_available = False
        self.hf_status = "disabled"

        if use_huggingface:
            try:
                from huggingface_hub import HfApi, InferenceClient
                if hf_api_token:
                    HfApi(token=hf_api_token).whoami()
                    self.client = InferenceClient(model=self.hf_model, token=hf_api_token)
                    self.hf_available = True
                    self.hf_status = f"ready ({hf_model})"
                    print(f"Initializing reasoning engine with HuggingFace API ({hf_model})...")
                else:
                    self.hf_status = "missing token"
                    print("HuggingFace API token not provided. Using heuristic-based reasoning...")
            except ImportError:
                self.hf_status = "huggingface_hub not installed"
                print("HuggingFace not available. Using heuristic-based reasoning...")
            except Exception as e:
                self.hf_status = f"unavailable: {e}"
                print(f"HuggingFace API error: {e}. Using heuristic-based reasoning...")
                self.hf_available = False

        self.generator = None
        self.package_rules = self._get_package_rules()

    def _call_huggingface(self, prompt: str, max_tokens: int = 150) -> str:
        """Call the configured HuggingFace Inference endpoint."""
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=self.hf_model, token=self.hf_api_token)
            response = client.text_generation(prompt, max_new_tokens=max_tokens)
            return response if response else ""
        except Exception as e:
            self.hf_status = f"generation failed: {e}"
            print(f"HuggingFace API error: {e}")
            return ""

    def _generate_llm_reasoning(self, class_info: Dict, target_type: str, class_context: str = "") -> str:
        """Generate reasoning using HuggingFace LLM"""
        if not self.hf_available:
            return ""

        class_name = class_info['class_name']
        annotations = class_info.get('annotations', []) or []
        methods = class_info.get('methods', []) or []
        fields = class_info.get('fields', []) or []
        extends = class_info.get('extends', '') or ''

        prompt = f"""You are a Java architecture expert. Analyze this class and explain why it belongs in a {target_type} package.

Class: {class_name}
Annotations: {', '.join(annotations) if annotations else 'none'}
Methods: {', '.join(methods[:10]) if methods else 'none'}
Fields: {', '.join(fields[:5]) if fields else 'none'}
Extends: {extends if extends else 'none'}
{class_context}

Provide a brief explanation (1-2 sentences) of why this class belongs in the {target_type} layer. Focus on architectural patterns, naming conventions, and annotations."""

        response = self._call_huggingface(prompt, max_tokens=150)
        if response:
            lines = response.split('\n')
            return ' '.join(lines[:3]).strip()
        return ""
        
    def _get_package_rules(self) -> Dict:
        """Rules for various architectural layers - more comprehensive patterns"""
        return {
            'model': {
                'keywords': ['entity', 'model', 'dto', 'pojo', 'bean', 'data', 'domain', 'request', 'response', 'VO', 'BO'],
                'annotations': ['Entity', 'Table', 'Data', 'NoArgsConstructor', 'AllArgsConstructor', 'Getter', 'Setter', 'Embeddable', 'Column', 'Id'],
                'patterns': [r'.*DTO$', r'.*Entity$', r'.*Model$', r'.*Request$', r'.*Response$', r'.*VO$', r'.*BO$', r'.*POJO$'],
                'method_patterns': ['get', 'set', 'is', 'validate', 'toString', 'equals', 'hashCode'],
                'extends_patterns': []
            },
            'service': {
                'keywords': ['service', 'logic', 'manager', 'handler', 'processor', 'provider', 'impl', 'business', 'facade'],
                'annotations': ['Service', 'Component', 'Transactional', 'Bean', 'Required'],
                'patterns': [r'.*Service$', r'.*Manager$', r'.*Processor$', r'.*Handler$', r'.*Impl$', r'.*Facade$'],
                'method_patterns': ['create', 'update', 'delete', 'find', 'get', 'process', 'execute', 'validate', 'calculate'],
                'extends_patterns': []
            },
            'controller': {
                'keywords': ['controller', 'rest', 'api', 'resource', 'endpoint', 'web', 'http', 'gateway'],
                'annotations': ['RestController', 'Controller', 'RequestMapping', 'GetMapping', 'PostMapping', 'PutMapping', 'DeleteMapping', 'PatchMapping', 'RequestBody', 'ResponseBody'],
                'patterns': [r'.*Controller$', r'.*Resource$', r'.*Endpoint$', r'.*API$', r'.*Handler$'],
                'method_patterns': ['get', 'post', 'put', 'delete', 'patch', 'request', 'handle'],
                'extends_patterns': []
            },
            'repository': {
                'keywords': ['repository', 'dao', 'persistence', 'mapper', 'crud', 'store', 'data'],
                'annotations': ['Repository', 'Mapper', 'EntityManager', 'Transactional'],
                'patterns': [r'.*Repository$', r'.*DAO$', r'.*Mapper$', r'.*Store$', r'.*Data$'],
                'method_patterns': ['find', 'save', 'delete', 'remove', 'update', 'insert', 'query', 'select', 'flush'],
                'extends_patterns': ['JpaRepository', 'CrudRepository', 'Repository']
            },
            'util': {
                'keywords': ['util', 'helper', 'utility', 'common', 'tools', 'formatter', 'parser', 'converter', 'validator'],
                'annotations': ['Component', 'Utility'],
                'patterns': [r'.*Util$', r'.*Helper$', r'.*Utils$', r'.*Formatter$', r'.*Parser$', r'.*Converter$', r'.*Validator$'],
                'method_patterns': ['convert', 'format', 'parse', 'validate', 'transform', 'encode', 'decode', 'encrypt', 'decrypt'],
                'extends_patterns': []
            },
            'config': {
                'keywords': ['config', 'configuration', 'settings', 'properties', 'setup', 'properties'],
                'annotations': ['Configuration', 'ConfigurationProperties', 'Bean', 'ConditionalOnProperty', 'PropertySource'],
                'patterns': [r'.*Config$', r'.*Configuration$', r'.*Settings$', r'.*Properties$'],
                'method_patterns': ['configure', 'initialize', 'load', 'setup', 'createBean'],
                'extends_patterns': []
            },
            'exception': {
                'keywords': ['exception', 'error', 'fault', 'failure', 'handler', 'throwable'],
                'annotations': ['ResponseStatus', 'ControllerAdvice', 'ExceptionHandler', 'Status'],
                'patterns': [r'.*Exception$', r'.*Error$', r'.*Fault$', r'.*RuntimeException$', r'.*Throwable$'],
                'method_patterns': [],
                'extends_patterns': ['Exception', 'RuntimeException', 'Throwable', 'Error']
            }
        }

    def identify_misplaced_classes(self, classes_data: List[Dict],
                                 dependency_graph: Dict,
                                 embeddings: Dict) -> List[str]:
        """Identify potentially misplaced classes using a weighted multi-factor scoring system"""
        misplaced = []

        # Pre-calculate class to package mapping for structural analysis
        class_to_pkg = {c['class_name']: c['package'] for c in classes_data}

        for class_info in classes_data:
            class_name = class_info['class_name']
            current_pkg = class_info['package'].lower()

            # 1. Calculate scores for all possible package types
            scores = {}
            components = {}
            for pkg_type in self.package_rules.keys():
                rule_s, sem_s, struct_s, total = self._calculate_component_scores(
                    class_info, pkg_type, dependency_graph, embeddings, class_to_pkg
                )
                scores[pkg_type] = total
                components[pkg_type] = (rule_s, sem_s, struct_s, total)

            # 2. Find the best fitting package type (by total score)
            best_pkg_type, best_score = max(scores.items(), key=lambda x: x[1])

            # Check for strong rule signals - if any type has very strong rule match, use that
            best_rule_type, (best_rule, _, _, _) = max(components.items(), key=lambda kv: kv[1][0])
            if best_rule >= 0.5:  # Lowered threshold from 0.75 to catch more cases
                best_pkg_type = best_rule_type
                best_score = scores[best_pkg_type]

            # 3. Determine if current package matches best type
            current_type = self._detect_pkg_type(current_pkg)
            current_score = scores.get(current_type, 0.0) if current_type != "unknown" else 0.0

            # 4. Decision logic - improved thresholds and logic
            if current_type != best_pkg_type:
                # Strong indicators that should always flag misplacement
                strong_rule_signal = best_rule >= 0.6
                # Annotation match is a very strong signal
                has_strong_annotation = self._has_strong_annotation_match(class_info, best_pkg_type)

                # Relaxed confidence requirements
                confident_enough = best_score >= 0.35  # Lowered from 0.50
                # Allow detection if the score difference is meaningful or current type is unknown
                margin_ok = (best_score - current_score) >= 0.05 if current_type != "unknown" else best_score >= 0.30

                if strong_rule_signal or has_strong_annotation or (confident_enough and margin_ok):
                    misplaced.append(class_name)

        return misplaced

    def _has_strong_annotation_match(self, class_info: Dict, target_type: str) -> bool:
        """Check if class has strong annotation matching target type"""
        annotations = class_info.get('annotations', []) or []
        target_annotations = self.package_rules[target_type]['annotations']

        # Count how many target-specific annotations are present
        matching = sum(1 for ann in annotations if ann in target_annotations)
        return matching >= 1  # At least one strong annotation match

    def suggest_target_packages(self, misplaced_classes: List[str],
                              classes_data: List[Dict],
                              dependency_graph: Dict,
                              embeddings: Dict) -> Dict[str, Dict]:
        """Generate detailed suggestions with reasoning"""
        suggestions = {}
        class_map = {c['class_name']: c for c in classes_data}
        class_to_pkg = {c['class_name']: c['package'] for c in classes_data}

        for name in misplaced_classes:
            class_info = class_map[name]

            # Re-calculate scores to find target
            comp = {
                pt: self._calculate_component_scores(class_info, pt, dependency_graph, embeddings, class_to_pkg)
                for pt in self.package_rules.keys()
            }

            # First try by total score
            target_type, (rule_s, _, _, confidence) = max(comp.items(), key=lambda kv: kv[1][3])

            # But if any rule score is strong, prefer that type (more deterministic)
            best_rule_type, (best_rule, _, _, best_total) = max(comp.items(), key=lambda kv: kv[1][0])
            if best_rule >= 0.5:  # Lowered from 0.75
                target_type = best_rule_type
                confidence = max(best_total, best_rule)

            # Generate reasoning text using template (LLM disabled due to hallucinations)
            reasoning = self._generate_reasoning(class_info, target_type, confidence, dependency_graph)

            # Keep original package base if available
            original_pkg = class_info.get('package', 'com.example')
            pkg_parts = original_pkg.rsplit('.', 1)
            base_pkg = pkg_parts[0] if len(pkg_parts) > 1 else 'com.example'

            suggestions[name] = {
                'suggested_package': f"{base_pkg}.{target_type}",
                'suggested_type': target_type,
                'confidence': confidence,
                'reasoning': reasoning,
                'method_used': 'Hybrid Structural-Semantic Reasoning'
            }

        return suggestions

    def _calculate_class_score(self, class_info: Dict, target_type: str, 
                             dependency_graph: Dict, embeddings: Dict,
                             class_to_pkg: Dict = None) -> float:
        """Weighted score for a class fitting into a package type"""
        return self._calculate_component_scores(class_info, target_type, dependency_graph, embeddings, class_to_pkg)[3]

    def _calculate_component_scores(
        self,
        class_info: Dict,
        target_type: str,
        dependency_graph: Dict,
        embeddings: Dict,
        class_to_pkg: Dict = None,
    ) -> tuple:
        """Return (rule_score, semantic_score, structural_score, total_score)."""
        embeddings = embeddings or {}
        rules = self.package_rules[target_type]
        name_lower = class_info['class_name'].lower()
        annotations = class_info.get("annotations", []) or []
        methods = class_info.get('methods', []) or []
        extends = class_info.get('extends', '') or ''
        implements = class_info.get('implements', []) or []

        # 1. Rule-based Score (Name, Annotations, Methods, Inheritance) - Weight: 0.5
        # Increased weight for rules as they are most reliable
        rule_score = 0.0

        # Pattern match on class name (strongest signal)
        if any(re.match(p.lower(), name_lower) for p in rules['patterns']):
            rule_score += 0.4

        # Keyword match (moderate signal)
        if any(k in name_lower for k in rules['keywords']):
            rule_score += 0.2

        # Annotation match (strong signal)
        matching_anns = [a for a in annotations if a in rules['annotations']]
        rule_score += min(len(matching_anns) * 0.25, 0.5)

        # Method pattern match
        method_patterns = rules.get('method_patterns', [])
        if method_patterns:
            method_matches = sum(1 for m in methods for mp in method_patterns if mp.lower() in m.lower())
            rule_score += min(method_matches * 0.05, 0.2)

        # Inheritance match (for repository, exception, etc.)
        extends_patterns = rules.get('extends_patterns', [])
        if extends_patterns and extends:
            if any(ext in extends for ext in extends_patterns):
                rule_score += 0.3
        if implements:
            for impl in implements:
                if any(ext in impl for ext in extends_patterns):
                    rule_score += 0.2

        rule_score = min(rule_score, 1.0)

        # 2. Semantic Score (Embeddings) - Weight: 0.25
        semantic_score = 0.5  # Default to neutral
        if 'class_embeddings' in embeddings and 'package_type_embeddings' in embeddings:
            class_vec = embeddings['class_embeddings'].get(class_info['class_name'])
            pkg_vec = embeddings['package_type_embeddings'].get(target_type)
            if class_vec is not None and pkg_vec is not None:
                semantic_score = self._cosine_similarity(class_vec, pkg_vec)

        # 3. Structural Score (Dependencies) - Weight: 0.25
        # Improved structural scoring - consider dependencies more carefully
        structural_score = 0.3  # Default to slightly below neutral
        if class_to_pkg and dependency_graph:
            deps = dependency_graph.get('class_dependencies', {}).get(class_info['class_name'], [])
            rev_deps = dependency_graph.get('reverse_dependencies', {}).get(class_info['class_name'], [])
            all_related = set(deps) | set(rev_deps)

            if all_related:
                matching_related = 0
                for related_class in all_related:
                    related_pkg = class_to_pkg.get(related_class, '').lower()
                    if target_type in related_pkg:
                        matching_related += 1

                structural_score = matching_related / len(all_related)

        total_score = (rule_score * 0.5) + (semantic_score * 0.25) + (structural_score * 0.25)
        return rule_score, semantic_score, structural_score, total_score

    def explain_class(self, class_info: Dict, dependency_graph: Dict, embeddings: Dict, classes_data: List[Dict]) -> Dict:
        """
        Debug helper: returns per-type component scores and the final decision context
        for a single class within a project.
        """
        class_to_pkg = {c["class_name"]: c["package"] for c in classes_data}
        scores = {}
        components = {}
        for pkg_type in self.package_rules.keys():
            rule_s, sem_s, struct_s, total = self._calculate_component_scores(
                class_info, pkg_type, dependency_graph, embeddings, class_to_pkg
            )
            scores[pkg_type] = total
            components[pkg_type] = {"rule": rule_s, "semantic": sem_s, "structural": struct_s, "total": total}

        best_total_type = max(scores.items(), key=lambda kv: kv[1])[0]
        best_rule_type = max(components.items(), key=lambda kv: kv[1]["rule"])[0]
        current_type = self._detect_pkg_type((class_info.get("package") or "").lower())

        return {
            "class_name": class_info.get("class_name"),
            "current_package": class_info.get("package"),
            "current_type": current_type,
            "best_by_total": best_total_type,
            "best_by_rule": best_rule_type,
            "scores": components,
        }

    def _generate_reasoning(self, class_info: Dict, target_type: str,
                          confidence: float, dependency_graph: Dict) -> str:
        """Generate reasoning using HuggingFace LLM or templates"""
        # Try HuggingFace first if available
        if self.hf_available:
            llm_reasoning = self._generate_llm_reasoning(class_info, target_type)
            if llm_reasoning and len(llm_reasoning) > 10:
                return llm_reasoning

        # Fall back to template-based reasoning
        reasons = []

        # Check name pattern match
        class_name = class_info['class_name'].lower()
        patterns = self.package_rules[target_type]['patterns']
        if any(re.match(p.lower(), class_name) for p in patterns):
            reasons.append(f"its name follows the standard {target_type} naming convention")

        # Check annotation match
        annotations = class_info.get('annotations', []) or []
        matching_anns = [a for a in annotations if a in self.package_rules[target_type]['annotations']]
        if matching_anns:
            reasons.append(f"it uses {target_type}-specific annotations ({', '.join(matching_anns[:2])})")

        # Check keyword match
        keywords = self.package_rules[target_type]['keywords']
        matching_kw = [k for k in keywords if k in class_name]
        if matching_kw:
            reasons.append(f"the class name contains '{matching_kw[0]}' which is typical for {target_type} components")

        # Check method patterns
        methods = class_info.get('methods', []) or []
        method_patterns = self.package_rules[target_type].get('method_patterns', [])
        if method_patterns:
            method_matches = [m for m in methods for mp in method_patterns if mp.lower() in m.lower()]
            if method_matches:
                reasons.append(f"its methods ({method_matches[0]}...) follow {target_type} patterns")

        # Check inheritance
        extends = class_info.get('extends', '') or ''
        if extends:
            extends_patterns = self.package_rules[target_type].get('extends_patterns', [])
            if any(ext in extends for ext in extends_patterns):
                reasons.append(f"it extends {target_type}-related class ({extends})")

        if not reasons:
            reasons.append("analysis of class structure and semantics indicates this layer is the best fit")

        return f"Based on architectural analysis, this class belongs in the {target_type} layer because " + ", and ".join(reasons) + "."

    def _detect_pkg_type(self, package_name: str) -> str:
        """Detect the likely type of a package from its name"""
        for pt in self.package_rules.keys():
            if pt in package_name:
                return pt
        return 'unknown'

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) if np.linalg.norm(a)*np.linalg.norm(b) > 0 else 0
