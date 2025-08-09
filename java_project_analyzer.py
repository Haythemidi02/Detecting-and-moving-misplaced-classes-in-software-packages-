"""
JavaProjectAnalyzer - Analyzes Java project structure and extracts class information
"""
import os # for file and directory operations
import re # regular expressions, used to search for patterns in Java code
from typing import List, Dict, Set # for type hinting
from pathlib import Path #  a modern way to work with file paths


class JavaProjectAnalyzer:
    def __init__(self):
        self.java_keywords = {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch',
            'char', 'class', 'const', 'continue', 'default', 'do', 'double',
            'else', 'enum', 'extends', 'final', 'finally', 'float', 'for',
            'goto', 'if', 'implements', 'import', 'instanceof', 'int',
            'interface', 'long', 'native', 'new', 'package', 'private',
            'protected', 'public', 'return', 'short', 'static', 'strictfp',
            'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
            'transient', 'try', 'void', 'volatile', 'while'
        }
    
    def analyze_project(self, project_path: str) -> List[Dict]:
        """Analyze Java project and extract class information"""
        classes_data = []
        java_files = self._find_java_files(project_path)
        
        for java_file in java_files:
            try:
                class_info = self._analyze_java_file(java_file, project_path)
                if class_info:
                    classes_data.extend(class_info)
            except Exception as e:
                print(f"Error analyzing {java_file}: {e}")
                continue
        
        return classes_data
    
    def _find_java_files(self, project_path: str) -> List[str]:
        """Find all Java files in the project"""
        java_files = []
        for root, dirs, files in os.walk(project_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {'.git', 'target', 'build', '.idea', 'out'}]
            
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
        
        return java_files
    
    def _analyze_java_file(self, file_path: str, project_root: str) -> List[Dict]:
        """Analyze a single Java file and extract class information"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        classes_info = []
        
        # Extract package
        package_match = re.search(r'package\s+([a-zA-Z0-9_.]+)\s*;', content)
        package_name = package_match.group(1) if package_match else 'default'
        
        # Extract imports
        imports = re.findall(r'import\s+([a-zA-Z0-9_.]+)\s*;', content)
        
        # Extract classes, interfaces, and enums
        class_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)'
        class_matches = re.findall(class_pattern, content)
        
        for class_name in class_matches:
            class_info = {
                'class_name': class_name,
                'package': package_name,
                'file_path': file_path,
                'relative_path': os.path.relpath(file_path, project_root),
                'imports': imports,
                'methods': self._extract_methods(content),
                'fields': self._extract_fields(content),
                'annotations': self._extract_annotations(content),
                'extends': self._extract_extends(content),
                'implements': self._extract_implements(content),
                'class_type': self._determine_class_type(content, class_name),
                'content_preview': content[:500].replace('\n', ' ').replace('\r', '')
            }
            classes_info.append(class_info)
        
        return classes_info
    
    def _extract_methods(self, content: str) -> List[str]:
        """Extract method signatures from Java content"""
        # Simplified method extraction
        method_pattern = r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?[A-Za-z0-9_<>\[\],\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)'
        return re.findall(method_pattern, content)
    
    def _extract_fields(self, content: str) -> List[str]:
        """Extract field names from Java content"""
        field_pattern = r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?[A-Za-z0-9_<>\[\],\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*[;=]'
        return [field for field in re.findall(field_pattern, content) if field not in self.java_keywords]
    
    def _extract_annotations(self, content: str) -> List[str]:
        """Extract annotations from Java content"""
        annotation_pattern = r'@([A-Za-z_][A-Za-z0-9_]*)'
        return re.findall(annotation_pattern, content)
    
    def _extract_extends(self, content: str) -> str:
        """Extract extends clause"""
        extends_match = re.search(r'extends\s+([A-Za-z_][A-Za-z0-9_]*)', content)
        return extends_match.group(1) if extends_match else ''
    
    def _extract_implements(self, content: str) -> List[str]:
        """Extract implements clause"""
        implements_match = re.search(r'implements\s+([A-Za-z0-9_,\s]+)', content)
        if implements_match:
            return [impl.strip() for impl in implements_match.group(1).split(',')]
        return []
    
    def _determine_class_type(self, content: str, class_name: str) -> str:
        """Determine if it's a class, interface, or enum"""
        if f'interface {class_name}' in content:
            return 'interface'
        elif f'enum {class_name}' in content:
            return 'enum'
        else:
            return 'class'