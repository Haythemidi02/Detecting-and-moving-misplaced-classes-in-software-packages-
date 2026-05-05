"""
JavaProjectAnalyzer - Analyzes Java project structure and extracts class information using javalang
"""
import os
import javalang
from typing import List, Dict, Set
from pathlib import Path


class JavaProjectAnalyzer:
    def __init__(self):
        pass
    
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
        """Analyze a single Java file and extract class information using javalang"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        try:
            tree = javalang.parse.parse(content)
        except Exception as e:
            # Fallback or log error
            raise ValueError(f"Javalang parsing failed: {e}")
        
        classes_info = []
        
        # Extract package
        package_name = tree.package.name if tree.package else 'default'
        
        # Extract imports
        imports = [imp.path for imp in tree.imports]
        
        # Extract classes, interfaces, and enums
        for type_decl in tree.types:
            class_name = type_decl.name
            
            # Extract methods
            methods = []
            for method in type_decl.methods:
                # Store more info about methods
                param_types = [p.type.name for p in method.parameters if hasattr(p.type, 'name')]
                return_type = method.return_type.name if method.return_type and hasattr(method.return_type, 'name') else 'void'
                methods.append({
                    'name': method.name,
                    'parameters': param_types,
                    'return_type': return_type,
                    'annotations': [ann.name for ann in method.annotations]
                })
            
            # Extract fields
            fields = []
            for field in type_decl.fields:
                field_type = field.type.name if hasattr(field.type, 'name') else 'unknown'
                for decl in field.declarators:
                    fields.append({
                        'name': decl.name,
                        'type': field_type
                    })
            
            # Extract annotations
            annotations = [ann.name for ann in type_decl.annotations]
            
            # Extract inheritance
            extends = type_decl.extends.name if hasattr(type_decl, 'extends') and type_decl.extends else ''
            implements = [impl.name for impl in type_decl.implements] if hasattr(type_decl, 'implements') and type_decl.implements else []
            
            # Determine class type
            class_type = 'class'
            if isinstance(type_decl, javalang.tree.InterfaceDeclaration):
                class_type = 'interface'
            elif isinstance(type_decl, javalang.tree.EnumDeclaration):
                class_type = 'enum'
            
            class_info = {
                'class_name': class_name,
                'package': package_name,
                'file_path': file_path,
                'relative_path': os.path.relpath(file_path, project_root),
                'imports': imports,
                'methods_detailed': methods,
                'methods': [m['name'] for m in methods],  # Backwards compatibility
                'fields_detailed': fields,
                'fields': [f['name'] for f in fields],    # Backwards compatibility
                'annotations': annotations,
                'extends': extends,
                'implements': implements,
                'class_type': class_type,
                'content_preview': content[:1000].replace('\n', ' ').replace('\r', '') # Increased preview
            }
            classes_info.append(class_info)
        
        return classes_info
