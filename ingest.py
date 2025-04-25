import os
import pickle
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np

class DocumentIngester:
    def __init__(self, data_dir: str = None, vector_store_dir: str = "vector_store"):
        if data_dir is None:
            # Always use backend/data relative to this file
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data_dir = data_dir
        self.vector_store_dir = vector_store_dir
        self.vectorizer = TfidfVectorizer()
        self.documents: List[str] = []
        self.document_metadata: List[Dict] = []
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(vector_store_dir, exist_ok=True)

    def load_documents(self):
        """Load documents from the data directory"""
        supported_extensions = {'.py', '.md', '.txt', '.json', '.sol'}
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            self.documents.append(content)
                            self.document_metadata.append({
                                'file_path': os.path.abspath(file_path),
                                'file_name': file
                            })
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")

    def create_embeddings(self):
        """Create TF-IDF embeddings and FAISS index"""
        if not self.documents:
            print("No documents loaded. Please load documents first.")
            return

        # Create TF-IDF embeddings
        tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        embeddings = tfidf_matrix.toarray().astype('float32')

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save the index and metadata
        faiss.write_index(index, os.path.join(self.vector_store_dir, "faiss_index.bin"))
        
        with open(os.path.join(self.vector_store_dir, "vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
        with open(os.path.join(self.vector_store_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.document_metadata, f)

    def process_documents(self):
        """Process all documents and create embeddings"""
        print("Loading documents...")
        self.load_documents()
        print(f"Loaded {len(self.documents)} documents")
        
        print("Creating embeddings...")
        self.create_embeddings()
        print("Embeddings created and saved successfully")

if __name__ == "__main__":
    ingester = DocumentIngester()
    ingester.process_documents() 