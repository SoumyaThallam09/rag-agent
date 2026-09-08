from pathlib import Path

file_path = Path("documents/support.txt")

text = file_path.read_text(encoding="utf-8")

# Split the document wherever there is a blank line
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)