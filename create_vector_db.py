import truststore
truststore.inject_into_ssl()


from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load document
file_path = Path("documents/support.txt")
text = file_path.read_text(encoding="utf-8")

# 2. Create chunks
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Create embeddings
embeddings = model.encode(chunks).tolist()

# 5. Create Chroma database
client = chromadb.PersistentClient(path="./chroma_db")

# 6. Create a collection
collection = client.get_or_create_collection(
    name="support_knowledge"
)

# 7. Store chunks + embeddings
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print(f"Stored {len(chunks)} chunks in Chroma.")