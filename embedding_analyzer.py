"""
EmbeddingAnalyzer - Computes semantic embeddings for classes using Hugging Face
"""
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch


class EmbeddingAnalyzer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a free Hugging Face model"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}
        
        # Common package types for comparison
        self.package_types = {
            'model': ['entity', 'data', 'domain', 'pojo', 'bean', 'dto'],
            'service': ['service', 'business', 'logic', 'manager', 'handler'],
            'controller': ['controller', 'rest', 'api', 'endpoint', 'web'],
            'repository': ['repository', 'dao', 'data', 'persistence', 'database'],
            'util': ['util', 'helper', 'common', 'utility', 'tools'],
            'config': ['config', 'configuration', 'settings', 'properties'],
            'exception': ['exception', 'error', 'fault', 'failure']
        }
    
    def compute_embeddings(self, classes_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Compute embeddings for all classes"""
        print("Computing embeddings for classes...")
        embeddings = {}
        
        for class_info in classes_data:
            class_name = class_info['class_name']
            
            # Create semantic representation of the class
            semantic_text = self._create_semantic_text(class_info)
            
            # Compute embedding
            embedding = self.model.encode(semantic_text, convert_to_tensor=False)
            embeddings[class_name] = embedding
        
        # Compute package type embeddings for comparison
        package_embeddings = self._compute_package_type_embeddings()
        
        return {
            'class_embeddings': embeddings,
            'package_type_embeddings': package_embeddings,
            'similarity_matrix': self._compute_similarity_matrix(embeddings),
            'clusters': self._perform_clustering(embeddings)
        }
    
    def _create_semantic_text(self, class_info: Dict) -> str:
        """Create semantic text representation of a class"""
        components = []
        
        # Class name (most important)
        components.append(f"Class name: {class_info['class_name']}")
        
        # Methods
        if class_info['methods']:
            methods_text = ' '.join(class_info['methods'][:10])  # Limit to avoid token limits
            components.append(f"Methods: {methods_text}")
        
        # Fields
        if class_info['fields']:
            fields_text = ' '.join(class_info['fields'][:10])
            components.append(f"Fields: {fields_text}")
        
        # Annotations
        if class_info['annotations']:
            annotations_text = ' '.join(class_info['annotations'])
            components.append(f"Annotations: {annotations_text}")
        
        # Class type
        components.append(f"Type: {class_info['class_type']}")
        
        # Content preview
        if class_info['content_preview']:
            components.append(f"Content: {class_info['content_preview']}")
        
        return ' '.join(components)
    
    def _compute_package_type_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute embeddings for different package types"""
        package_embeddings = {}
        
        for package_type, keywords in self.package_types.items():
            text = f"{package_type} package containing {' '.join(keywords)} related classes"
            embedding = self.model.encode(text, convert_to_tensor=False)
            package_embeddings[package_type] = embedding
        
        return package_embeddings
    
    def _compute_similarity_matrix(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """Compute similarity matrix between classes"""
        class_names = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[name] for name in class_names])
        
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        return {
            'class_names': class_names,
            'matrix': similarity_matrix
        }
    
    def _perform_clustering(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """Perform clustering on class embeddings"""
        if len(embeddings) < 2:
            return {'clusters': {}, 'n_clusters': 0}
        
        embedding_matrix = np.array(list(embeddings.values()))
        class_names = list(embeddings.keys())
        
        # Determine optimal number of clusters (max 10)
        n_clusters = min(max(2, len(embeddings) // 3), 10)
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding_matrix)
            
            # Group classes by cluster
            clusters = {}
            for i, class_name in enumerate(class_names):
                cluster_id = int(cluster_labels[i])
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(class_name)
            
            return {
                'clusters': clusters,
                'n_clusters': n_clusters,
                'cluster_centers': kmeans.cluster_centers_
            }
        except Exception as e:
            print(f"Clustering failed: {e}")
            return {'clusters': {}, 'n_clusters': 0}
    
    def find_most_similar_package_type(self, class_embedding: np.ndarray, 
                                     package_type_embeddings: Dict[str, np.ndarray]) -> tuple:
        """Find the most similar package type for a given class embedding"""
        similarities = {}
        
        for package_type, package_embedding in package_type_embeddings.items():
            similarity = cosine_similarity(
                class_embedding.reshape(1, -1), 
                package_embedding.reshape(1, -1)
            )[0][0]
            similarities[package_type] = similarity
        
        best_match = max(similarities.items(), key=lambda x: x[1])
        return best_match[0], best_match[1]
    
    def compute_class_similarity(self, class1_embedding: np.ndarray, 
                               class2_embedding: np.ndarray) -> float:
        """Compute similarity between two class embeddings"""
        return cosine_similarity(
            class1_embedding.reshape(1, -1),
            class2_embedding.reshape(1, -1)
        )[0][0]