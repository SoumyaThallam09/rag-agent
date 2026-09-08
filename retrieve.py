import truststore
truststore.inject_into_ssl()

from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to our existing Chroma database
client = chromadb.PersistentClient(path="./chroma_db")


collection = client.get_collection(
    name="support_knowledge")

# Ask a question
question = "What can cause a migration to become incomplete?"

# Convert question into an embedding
question_embedding = model.encode([question]).tolist()

# Search the vector database
results = collection.query(
    query_embeddings=question_embedding,
    n_results=2
)

print("\nQuestion:")
print(question)

print("\nRetrieved chunks:")

for document in results["documents"][0]:
    print("\n---")
    print(document)