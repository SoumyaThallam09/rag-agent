import truststore
truststore.inject_into_ssl()

import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print("API key loaded:", api_key is not None)

client = Groq(api_key=api_key)

# Read the migration knowledge base
document_path = os.path.join("documents", "support.txt")

with open(document_path, "r", encoding="utf-8") as file:
    knowledge_base = file.read()

print("Knowledge base loaded successfully.")

while True:
    question = input("\nAsk your question: ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    prompt = f"""
    You are a Zephyr Migration Support Assistant.

    Answer the user's question using ONLY the information contained in
    the KNOWLEDGE BASE below.

    IMPORTANT RULES:

    1. Understand the meaning of the user's question, including paraphrased
   or differently worded questions.

    2. Look for information with the same meaning, even if the exact words
   used by the user do not appear in the knowledge base.

    3. Do NOT require an exact keyword match.

    4. Do NOT use general knowledge or information outside the knowledge base.

    5. If the knowledge base states that something is not supported,
   clearly answer that it is not supported.

    6. Only say "I could not find this information in the available migration
   documentation" if there is genuinely no information related to
   the user's question.

    KNOWLEDGE BASE:
----------------
{knowledge_base}
----------------

USER QUESTION:
{question}

Provide a clear and concise answer based on the knowledge base.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful Zephyr Migration Support Assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nAnswer:")
    print(response.choices[0].message.content)