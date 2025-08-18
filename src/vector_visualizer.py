import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from langchain_mongodb import MongoDBAtlasVectorSearch

class VectorVisualizer:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def _get_mongodb_data(self):
        """Extract data from MongoDB vectorstore - only doc_type and content needed"""
        try:
            if isinstance(self.vectorstore, MongoDBAtlasVectorSearch):
                collection = self.vectorstore.collection
                
                # Only get the fields we actually need for visualization
                cursor = collection.find(
                    {}, 
                    {
                        "content": 1, 
                        "embedding": 1, 
                        "doc_type": 1
                    }
                )
                
                docs = list(cursor)
                
                if not docs:
                    print(f"[WARNING] No documents found for visualization")
                    return None, None, None
                
                
                
                # Extract only what we need
                vectors = np.array([doc["embedding"] for doc in docs])
                documents = [doc.get("content", "")[:200] for doc in docs]  # Truncated text for hover
                doc_types = [doc.get('doc_type', 'unknown') for doc in docs]  # Just the doc_type
                
                print(f"[DEBUG] Found doc_types: {set(doc_types)}")
                print(f"[INFO] Loaded {len(vectors)} vectors for visualization")
                
                return vectors, documents, doc_types
            else:
                print(f"[ERROR] Unsupported vectorstore type")
                return None, None, None
                
        except Exception as e:
            print(f"[ERROR] Failed to get MongoDB data: {e}")
            return None, None, None


    def visualize_2d(self):
        """Create 2D visualization - no user_id parameter needed"""
        vectors, documents, doc_types = self._get_mongodb_data()
        
        if vectors is None or len(vectors) < 2:
            print("[WARNING] Need at least 2 documents for visualization")
            return None
            
        
        # Dynamic color mapping
        unique_types = list(set(doc_types))
        print(f"[INFO] Document types found: {unique_types}")
        color_palette = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_map = {doc_type: color_palette[i % len(color_palette)] for i, doc_type in enumerate(unique_types)}
        colors = [color_map[t] for t in doc_types]


        # Reduce dimensionality
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1))
        reduced_vectors = tsne.fit_transform(vectors)

        def wrap_text(text, width=80):
            return '<br>'.join([text[i:i+width] for i in range(0, len(text), width)])

        # Create plot
        fig = go.Figure(data=[go.Scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            mode='markers',
            marker=dict(size=8, color=colors, opacity=0.8),
            text=[f"Type: {t}<br>Text: {wrap_text(d)}" for t, d in zip(doc_types, documents)],
            hoverinfo='text',
            hoverlabel=dict(font=dict(size=10))
        )])
        
        # Get collection name for title
        collection_name = self.vectorstore.collection.name if hasattr(self.vectorstore, 'collection') else 'User Collection'
        
        fig.update_layout(
            title=f'2D Vector Visualization - {collection_name}',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            width=800,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40)
        )
        fig.show()
        return fig
        
    def visualize_3d(self):
        """Create 3D visualization - no user_id parameter needed"""
        vectors, documents, doc_types = self._get_mongodb_data()
        
        if vectors is None or len(vectors) < 2:
            print("[WARNING] Need at least 2 documents for visualization")
            return None
        
        # Dynamic color mapping
        unique_types = list(set(doc_types))
        color_palette = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_map = {doc_type: color_palette[i % len(color_palette)] for i, doc_type in enumerate(unique_types)}
        colors = [color_map[t] for t in doc_types]

        # Reduce dimensionality
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(vectors)-1))
        reduced_vectors = tsne.fit_transform(vectors)
        
        # Create 3D plot
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode='markers',
            marker=dict(size=8, color=colors, opacity=0.8),
            text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
            hoverinfo='text'
        )])
        
        # Get collection name for title
        collection_name = self.vectorstore.collection.name if hasattr(self.vectorstore, 'collection') else 'User Collection'
        
        fig.update_layout(
            title=f'3D Vector Visualization - {collection_name}',
            scene=dict(xaxis_title='t-SNE 1', yaxis_title='t-SNE 2', zaxis_title='t-SNE 3'),
            width=900,
            height=700,
            margin=dict(r=20, b=10, l=10, t=40)
        )
        fig.show()
        return fig
    
   