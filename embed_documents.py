from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load the document
file_path = Path("documents/support.txt")
text = file_path.read_text(encoding="utf-8")

# Split into chunks
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert chunks into embeddings
embeddings = model.encode(chunks)

print(f"Number of chunks: {len(chunks)}")
print(f"Embedding shape: {embeddings.shape}")

for i, embedding in enumerate(embeddings):
    print(f"\nChunk {i + 1}")
    print(f"Text: {chunks[i]}")
    print(f"Vector length: {len(embedding)}")