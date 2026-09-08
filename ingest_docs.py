import truststore
truststore.inject_into_ssl()

from pathlib import Path
import hashlib

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb


# =========================================================
# CONFIGURATION
# =========================================================

SUPPORT_FILE = Path("documents/support.txt")

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "support_knowledge"

URLS = [
    "https://support.smartbear.com/zephyr/docs/en/zephyr-editions-feature-comparison.html",

    "https://support.smartbear.com/zephyr/docs/en/zephyr-squad-to-zephyr-upgrade-guide.html",

    "https://support.smartbear.com/zephyr/docs/en/zephyr-squad-to-zephyr-upgrade-guide/data-transfer.html"

    ]


# =========================================================
# READ SMARTBEAR WEBPAGE
# =========================================================

def get_webpage_content(url):

    try:

        response = requests.get(
            url,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary content
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        # Extract text
        text = soup.get_text(
            separator="\n",
            strip=True
        )

        return text

    except Exception as e:

        print(f"Error reading {url}: {e}")

        return ""


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(text):

    return [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]


# =========================================================
# CREATE UNIQUE ID
# =========================================================

def create_id(source, chunk):

    value = f"{source}:{chunk}"

    return hashlib.md5(
        value.encode("utf-8")
    ).hexdigest()


# =========================================================
# COLLECT ALL KNOWLEDGE
# =========================================================

all_chunks = []
all_ids = []
all_metadatas = []


# =========================================================
# 1. READ support.txt
# =========================================================

print("\nReading support.txt...")

support_text = SUPPORT_FILE.read_text(
    encoding="utf-8"
)

support_chunks = create_chunks(
    support_text
)

print(
    f"Found {len(support_chunks)} chunks in support.txt."
)


for chunk in support_chunks:

    all_chunks.append(chunk)

    all_ids.append(
        create_id(
            "support.txt",
            chunk
        )
    )

    all_metadatas.append(
        {
            "source": "support.txt"
        }
    )


# =========================================================
# 2. READ SMARTBEAR DOCUMENTATION
# =========================================================

for url in URLS:

    print(f"\nReading SmartBear documentation:")
    print(url)

    content = get_webpage_content(url)

    if not content:

        print("No content retrieved. Skipping.")

        continue

    chunks = create_chunks(
        content
    )

    print(
        f"Found {len(chunks)} chunks."
    )


    for chunk in chunks:

        all_chunks.append(chunk)

        all_ids.append(
            create_id(
                url,
                chunk
            )
        )

        all_metadatas.append(
            {
                "source": "SmartBear documentation",
                "url": url
            }
        )


# =========================================================
# CHECK WHETHER KNOWLEDGE WAS FOUND
# =========================================================

if not all_chunks:

    print(
        "\nNo knowledge was found. "
        "ChromaDB was not updated."
    )

    exit()


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print(
    "\nLoading embedding model..."
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

print(
    "Creating embeddings..."
)

embeddings = model.encode(
    all_chunks
).tolist()


# =========================================================
# CONNECT TO CHROMADB
# =========================================================

print(
    "\nConnecting to ChromaDB..."
)

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# =========================================================
# GET COLLECTION
# =========================================================

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================================================
# UPDATE CHROMADB
# =========================================================

print(
    "Updating ChromaDB..."
)

collection.upsert(
    ids=all_ids,
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadatas
)


# =========================================================
# COMPLETION
# =========================================================

print("\n========================================")
print("INGESTION COMPLETED SUCCESSFULLY")
print("========================================")

print(
    f"Total chunks processed: {len(all_chunks)}"
)

print(
    f"ChromaDB collection: {COLLECTION_NAME}"
)

print(
    f"ChromaDB path: {CHROMA_PATH}"
)

print(
    f"Total chunks currently in collection: "
    f"{collection.count()}"
)