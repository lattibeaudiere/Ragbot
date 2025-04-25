import aiohttp
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

print("Testing imports...")
print("✅ aiohttp imported successfully")
print("✅ faiss imported successfully")
print("✅ numpy imported successfully")
print("✅ scikit-learn imported successfully")
print("✅ python-dotenv imported successfully")

# Test numpy functionality
arr = np.array([1, 2, 3])
print(f"\nTesting numpy: {arr}")

# Test scikit-learn functionality
vectorizer = TfidfVectorizer()
print("\n✅ TfidfVectorizer created successfully")

print("\nAll dependencies are working correctly!") 