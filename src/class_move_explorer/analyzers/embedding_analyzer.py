"""
EmbeddingAnalyzer - Computes semantic embeddings for classes using Sentence Transformers
"""
import numpy as np
from typing import Dict, List
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch


class EmbeddingAnalyzer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a lightweight but effective sentence transformer model"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Package types for semantic comparison
        self.package_type_definitions = {
            'model': 'Data structures, entities, DTOs, and domain objects that represent the application data.',
            'service': 'Business logic, processing services, and managers that handle application workflows.',
            'controller': 'Web endpoints, REST controllers, and API resources that handle HTTP requests.',
            'repository': 'Data access objects, persistence logic, and database interaction layers.',
            'util': 'Utility classes, helpers, and common tools used across the application.',
            'config': 'Configuration classes, settings, and application setup properties.',
            'exception': 'Custom exceptions, error handlers, and fault management classes.'
        }
    
    def compute_embeddings(self, classes_data: List[Dict]) -> Dict:
        """Compute embeddings for all classes and package types"""
        print(f"Computing embeddings for {len(classes_data)} classes...")
        
        class_embeddings = {}
        for class_info in classes_data:
            class_name = class_info['class_name']
            semantic_text = self._create_semantic_text(class_info)
            embedding = self.model.encode(semantic_text, convert_to_tensor=False)
            class_embeddings[class_name] = embedding
        
        # Compute embeddings for package definitions
        package_type_embeddings = {}
        for pkg_type, definition in self.package_type_definitions.items():
            package_type_embeddings[pkg_type] = self.model.encode(definition, convert_to_tensor=False)
        
        return {
            'class_embeddings': class_embeddings,
            'package_type_embeddings': package_type_embeddings,
            'similarity_matrix': self._compute_similarity_matrix(class_embeddings),
            'clusters': self._perform_clustering(class_embeddings)
        }
    
    def _create_semantic_text(self, class_info: Dict) -> str:
        """Create a rich semantic description of the class for embedding generation"""
        components = []
        
        # 1. Identity
        components.append(f"Class: {class_info['class_name']}")
        components.append(f"Type: {class_info['class_type']}")
        
        # 2. Context (Annotations often reveal intent)
        if class_info['annotations']:
            anns = ', '.join(class_info['annotations'])
            components.append(f"Role/Annotations: {anns}")
        
        # 3. Behavior (Method names indicate functionality)
        if class_info['methods']:
            methods = ', '.join(class_info['methods'][:15])
            components.append(f"Handles actions like: {methods}")
        
        # 4. State (Fields indicate what it manages)
        if class_info['fields']:
            fields = ', '.join(class_info['fields'][:10])
            components.append(f"Manages data: {fields}")
        
        # 5. Relationships
        if class_info['extends']:
            components.append(f"Inherits from: {class_info['extends']}")
        if class_info['implements']:
            components.append(f"Implements: {', '.join(class_info['implements'])}")
            
        return " | ".join(components)
    
    def _compute_similarity_matrix(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """Compute similarity matrix between all classes"""
        names = list(embeddings.keys())
        if not names:
            return {'class_names': [], 'matrix': []}
            
        vectors = np.array([embeddings[name] for name in names])
        sim_matrix = cosine_similarity(vectors)
        
        return {
            'class_names': names,
            'matrix': sim_matrix.tolist()
        }
    
    def _perform_clustering(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """Cluster classes based on semantic similarity to find latent groupings"""
        if len(embeddings) < 3:
            return {'clusters': {}, 'n_clusters': 0}
            
        vectors = np.array(list(embeddings.values()))
        names = list(embeddings.keys())
        
        # Heuristic for number of clusters
        n_clusters = min(len(embeddings) // 2, 8) 
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[int(label)].append(names[idx])
            
        return {
            'clusters': dict(clusters),
            'n_clusters': n_clusters
        }

    def get_most_similar_package_type(self, class_embedding: np.ndarray, 
                                   pkg_embeddings: Dict[str, np.ndarray]) -> tuple:
        """Find which predefined package type best fits this class embedding"""
        best_type = 'unknown'
        max_sim = -1.0
        
        class_vec = class_embedding.reshape(1, -1)
        
        for pkg_type, pkg_vec in pkg_embeddings.items():
            sim = cosine_similarity(class_vec, pkg_vec.reshape(1, -1))[0][0]
            if sim > max_sim:
                max_sim = sim
                best_type = pkg_type
                
        return best_type, max_sim
