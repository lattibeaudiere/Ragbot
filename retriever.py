import os
import pickle
import faiss
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer

class DocumentRetriever:
    def __init__(self, vector_store_dir: str = "vector_store"):
        self.vector_store_dir = vector_store_dir
        self.index = None
        self.vectorizer = None
        self.document_metadata = None
        self.load_artifacts()

    def load_artifacts(self):
        """Load the FAISS index, vectorizer, and metadata"""
        try:
            # Load FAISS index
            index_path = os.path.join(self.vector_store_dir, "faiss_index.bin")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            else:
                print("FAISS index not found. Retrieval will be unavailable until files are ingested.")
                self.index = None

            # Load vectorizer
            vectorizer_path = os.path.join(self.vector_store_dir, "vectorizer.pkl")
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            else:
                self.vectorizer = None

            # Load metadata
            metadata_path = os.path.join(self.vector_store_dir, "metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
            else:
                self.document_metadata = None

        except Exception as e:
            print(f"Error loading artifacts: {str(e)}")
            self.index = None
            self.vectorizer = None
            self.document_metadata = None

    def retrieve_documents(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve the k most relevant documents for a given query
        Returns a list of dictionaries containing document content and metadata
        """
        if not self.index or not self.vectorizer or not self.document_metadata:
            print("Vector store not initialized. No documents available for retrieval.")
            return []

        # Convert query to embedding
        query_embedding = self.vectorizer.transform([query]).toarray().astype('float32')

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)

        # Get relevant documents
        relevant_docs = []
        for idx in indices[0]:
            if idx < len(self.document_metadata):
                doc_info = self.document_metadata[idx]
                try:
                    # Ensure absolute path for file reading
                    file_path = doc_info['file_path']
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        relevant_docs.append({
                            'content': content,
                            'file_name': doc_info['file_name'],
                            'file_path': doc_info['file_path']
                        })
                except Exception as e:
                    print(f"Error reading document {doc_info['file_path']}: {str(e)}")

        return relevant_docs

if __name__ == "__main__":
    # Example usage
    retriever = DocumentRetriever()
    query = "example query"
    results = retriever.retrieve_documents(query)
    for doc in results:
        print(f"File: {doc['file_name']}")
        print(f"Content: {doc['content'][:200]}...")  # Print first 200 chars
        print("---") 