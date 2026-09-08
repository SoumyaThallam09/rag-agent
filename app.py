import truststore
truststore.inject_into_ssl()

import os
import re
import html
import requests
import numpy as np
import streamlit as st

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Zephyr Cloud Migration Assistant",
    page_icon="🔍",
    layout="centered"
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=api_key)


# =========================================================
# LOGO PATH
# =========================================================

logo_path = os.path.join(
    "images",
    "zephyr-logo.png"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .zephyr-header {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        background: linear-gradient(
            135deg,
            #f5f7fa,
            #ffffff
        );
        border: 1px solid #e5e7eb;
    }

    .header-title {
        font-size: 27px;
        font-weight: 700;
        color: #172033;
        line-height: 1.25;
    }

    .header-subtitle {
        font-size: 15px;
        color: #667085;
        margin-top: 6px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-top: 15px;
    }

    .source-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-top: 10px;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 12px;
        margin-top: 35px;
        padding-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

header_col1, header_col2 = st.columns(
    [1, 5],
    vertical_alignment="center"
)

with header_col1:

    if os.path.exists(logo_path):

        st.image(
            logo_path,
            width=85
        )

with header_col2:

    st.markdown(
        """
        <div class="header-title">
            Zephyr Squad Cloud → Essential / Standard / Advanced Cloud Migration Assistant
        </div>

        <div class="header-subtitle">
            AI-powered migration support
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MANUAL KNOWLEDGE BASE REFRESH
# =========================================================

if st.sidebar.button("🔄 Refresh Knowledge Base"):

    st.cache_data.clear()
    st.cache_resource.clear()

    st.rerun()


# =========================================================
# LOCAL KNOWLEDGE BASE
# =========================================================

LOCAL_KB_FOLDER = "documents"


def get_kb_folder_signature():

    if not os.path.isdir(LOCAL_KB_FOLDER):
        return ()

    signature = []

    for filename in sorted(
        os.listdir(LOCAL_KB_FOLDER)
    ):

        if not filename.lower().endswith(".txt"):
            continue

        file_path = os.path.join(
            LOCAL_KB_FOLDER,
            filename
        )

        if os.path.isfile(file_path):

            signature.append(
                (
                    filename,
                    os.path.getmtime(file_path)
                )
            )

    return tuple(signature)


@st.cache_data
def load_local_knowledge_base_files(
    kb_folder_signature
):

    documents = []

    if not os.path.isdir(
        LOCAL_KB_FOLDER
    ):
        return documents

    for filename, _ in kb_folder_signature:

        file_path = os.path.join(
            LOCAL_KB_FOLDER,
            filename
        )

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read()

        except Exception:

            continue

        if content.strip():

            documents.append(
                {
                    "source": f"Local Knowledge Base: {filename}",
                    "url": "",
                    "content": content
                }
            )

    return documents


kb_signature = get_kb_folder_signature()

local_kb_documents = (
    load_local_knowledge_base_files(
        kb_signature
    )
)


# =========================================================
# SMARTBEAR DOCUMENTATION
# =========================================================

DOCUMENTATION_URLS = [

    "https://support.smartbear.com/zephyr/docs/en/zephyr-editions-feature-comparison.html",

    "https://support.smartbear.com/zephyr/docs/en/zephyr-squad-to-zephyr-upgrade-guide.html",

    "https://support.smartbear.com/zephyr/docs/en/zephyr-squad-to-zephyr-upgrade-guide/data-transfer.html"

]


# =========================================================
# FETCH WEB PAGE
# =========================================================

@st.cache_data(ttl=3600)
def fetch_webpage(url):

    try:

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/151.0 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "noscript"
            ]
        ):

            tag.decompose()

        content = []

        # -------------------------------------------------
        # HEADINGS
        # -------------------------------------------------

        for heading in soup.find_all(
            ["h1", "h2", "h3", "h4"]
        ):

            heading_text = heading.get_text(
                " ",
                strip=True
            )

            if heading_text:

                content.append(
                    f"\nSECTION: {heading_text}\n"
                )

        # -------------------------------------------------
        # PARAGRAPHS
        # -------------------------------------------------

        for paragraph in soup.find_all("p"):

            text = paragraph.get_text(
                " ",
                strip=True
            )

            if text:

                content.append(text)

        # -------------------------------------------------
        # LIST ITEMS
        # -------------------------------------------------

        for item in soup.find_all("li"):

            text = item.get_text(
                " ",
                strip=True
            )

            if text:

                content.append(
                    f"- {text}"
                )

        # -------------------------------------------------
        # TABLES
        # -------------------------------------------------

        for table in soup.find_all("table"):

            rows = table.find_all("tr")

            if not rows:
                continue

            headers = []

            first_row_cells = rows[0].find_all(
                ["th", "td"]
            )

            for cell in first_row_cells:

                headers.append(
                    cell.get_text(
                        " ",
                        strip=True
                    )
                )

            for row in rows[1:]:

                cells = row.find_all(
                    ["th", "td"]
                )

                if not cells:
                    continue

                values = []

                for cell in cells:

                    values.append(
                        cell.get_text(
                            " ",
                            strip=True
                        )
                    )

                row_content = []

                for index, value in enumerate(values):

                    if not value:
                        continue

                    if index < len(headers):

                        row_content.append(
                            f"{headers[index]}: {value}"
                        )

                    else:

                        row_content.append(value)

                if row_content:

                    content.append(
                        "\nTABLE ROW:\n"
                        +
                        "\n".join(row_content)
                    )

        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------

        if not content:

            return soup.get_text(
                separator="\n",
                strip=True
            )

        return "\n".join(content)

    except Exception as e:

        return (
            f"ERROR FETCHING {url}: "
            f"{str(e)}"
        )


# =========================================================
# LOAD ALL DOCUMENTATION
# =========================================================

@st.cache_data(ttl=3600)
def load_all_documentation(
    kb_signature,
    kb_documents
):

    documents = list(
        kb_documents
    )

    for url in DOCUMENTATION_URLS:

        content = fetch_webpage(url)

        if content.startswith(
            "ERROR FETCHING"
        ):

            continue

        documents.append(
            {
                "source": "SmartBear Documentation",
                "url": url,
                "content": content
            }
        )

    return documents


all_documents = load_all_documentation(
    kb_signature,
    local_kb_documents
)


# =========================================================
# CHUNK DOCUMENTATION
# =========================================================

def create_chunks(
    text,
    source,
    url,
    chunk_size=400,
    overlap=50
):

    words = text.split()

    chunks = []

    start = 0

    current_section = "GENERAL"

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk_words = words[start:end]

        chunk_text = " ".join(
            chunk_words
        )

        section_matches = re.findall(
            r"SECTION:\s*(.*?)(?=\s+SECTION:|$)",
            chunk_text,
            flags=re.IGNORECASE
        )

        if section_matches:

            current_section = (
                section_matches[-1]
                .strip()
            )

        chunks.append(
            {
                "text": chunk_text,
                "source": source,
                "url": url,
                "section": current_section
            }
        )

        if end >= len(words):

            break

        start = end - overlap

    return chunks


# =========================================================
# BUILD ALL CHUNKS
# =========================================================

@st.cache_data(ttl=3600)
def build_all_chunks(
    kb_signature,
    documents
):

    all_chunks = []

    for document in documents:

        document_chunks = create_chunks(
            text=document["content"],
            source=document["source"],
            url=document["url"],
            chunk_size=400,
            overlap=50
        )

        all_chunks.extend(
            document_chunks
        )

    return all_chunks


chunks = build_all_chunks(
    kb_signature,
    all_documents
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = (
    load_embedding_model()
)


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

@st.cache_resource
def create_embeddings(
    chunk_texts
):

    embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    embeddings = (
        embeddings / norms
    )

    return embeddings


chunk_texts = tuple(
    chunk["text"]
    for chunk in chunks
)


if chunk_texts:

    chunk_embeddings = (
        create_embeddings(
            chunk_texts
        )
    )

else:

    chunk_embeddings = np.array([])


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# RETRIEVE RELEVANT DOCUMENTS
# =========================================================

def retrieve_documents(
    question,
    top_k=5
):

    if len(chunks) == 0:
        return []

    # -------------------------------------------------
    # QUESTION EMBEDDING
    # -------------------------------------------------

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )[0]

    norm = np.linalg.norm(
        question_embedding
    )

    if norm != 0:
        question_embedding = (
            question_embedding / norm
        )

    similarities = np.dot(
        chunk_embeddings,
        question_embedding
    )

    # -------------------------------------------------
    # NORMALIZE QUESTION
    # -------------------------------------------------

    normalized_question = normalize_text(
        question
    )

    question_tokens = (
        normalized_question.split()
    )

    # -------------------------------------------------
    # STOP WORDS
    # -------------------------------------------------

    stop_words = {
        "what",
        "where",
        "when",
        "which",
        "who",
        "how",
        "why",
        "is",
        "are",
        "was",
        "were",
        "does",
        "do",
        "will",
        "can",
        "could",
        "would",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "from",
        "with",
        "and",
        "or",
        "this",
        "that",
        "be",
        "it"
    }

    meaningful_question_words = (
        set(question_tokens) - stop_words
    )

    # -------------------------------------------------
    # DETECT QUESTION TYPE
    #
    # This is intent classification, NOT an answer.
    # -------------------------------------------------

    migration_terms = {
        "migrate",
        "migrated",
        "migration",
        "transfer",
        "transferred",
        "upgrade",
        "upgraded",
        "move",
        "moved",
        "import",
        "export",
        "data",
        "source",
        "destination",
        "mapping",
        "mapped",
        "history",
        "audit",
        "legacy",
        "squad"
    }

    migration_question = bool(
        meaningful_question_words.intersection(
            migration_terms
        )
    )

    scored_chunks = []

    for index, chunk in enumerate(chunks):

        chunk_text = chunk["text"]

        normalized_chunk = normalize_text(
            chunk_text
        )

        chunk_words = set(
            normalized_chunk.split()
        )

        # -------------------------------------------------
        # SEMANTIC SCORE
        # -------------------------------------------------

        semantic_score = float(
            similarities[index]
        )

        # -------------------------------------------------
        # KEYWORD OVERLAP
        # -------------------------------------------------

        keyword_overlap = len(
            meaningful_question_words.intersection(
                chunk_words
            )
        )

        keyword_score = min(
            keyword_overlap * 0.025,
            0.20
        )

        # -------------------------------------------------
        # EXACT PHRASE MATCH
        # -------------------------------------------------

        phrase_score = 0

        if (
            normalized_question
            and
            normalized_question in normalized_chunk
        ):

            phrase_score += 0.30

        # -------------------------------------------------
        # BIGRAMS
        # -------------------------------------------------

        if len(question_tokens) >= 2:

            bigrams = [
                " ".join(
                    question_tokens[i:i + 2]
                )
                for i in range(
                    len(question_tokens) - 1
                )
            ]

            for phrase in bigrams:

                if phrase in normalized_chunk:

                    phrase_score += 0.05

        # -------------------------------------------------
        # TRIGRAMS
        # -------------------------------------------------

        if len(question_tokens) >= 3:

            trigrams = [
                " ".join(
                    question_tokens[i:i + 3]
                )
                for i in range(
                    len(question_tokens) - 2
                )
            ]

            for phrase in trigrams:

                if phrase in normalized_chunk:

                    phrase_score += 0.08

        phrase_score = min(
            phrase_score,
            0.25
        )

        # -------------------------------------------------
        # SECTION MATCH
        # -------------------------------------------------

        section_score = 0

        section_text = normalize_text(
            chunk.get(
                "section",
                ""
            )
        )

        if section_text:

            section_words = set(
                section_text.split()
            )

            section_overlap = len(
                meaningful_question_words.intersection(
                    section_words
                )
            )

            section_score = min(
                section_overlap * 0.03,
                0.10
            )

        # -------------------------------------------------
        # DOCUMENT TYPE / SOURCE PRIORITY
        #
        # This does NOT contain an answer.
        # It only tells retrieval which documentation
        # is more relevant to the question.
        # -------------------------------------------------

        source_priority = 0.0

        if migration_question:
            if chunk["url"] == "":
                source_priority = 0.50
            elif "squad-to-zephyr-upgrade-guide" in chunk["url"]:
                source_priority = 0.20
            elif "feature-comparison" in chunk["url"]:
                source_priority = 0.0

        # -------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------

        final_score = (
            semantic_score
            + keyword_score
            + phrase_score
            + section_score
            + source_priority
        )

        scored_chunks.append(
            (
                final_score,
                semantic_score,
                chunk["url"],
                chunk.get("section", ""),
                index,
                chunk
            )
        )

    # -------------------------------------------------
    # DETERMINISTIC SORT
    #
    # If two chunks have the same score, the ordering
    # remains stable.
    # -------------------------------------------------

    scored_chunks.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2],
            item[3],
            item[4]
        )
    )

    return [
        {
            "score": item[0],
            "chunk": item[5]
        }
        for item in scored_chunks[:top_k]
    ]       

# =========================================================
# BUILD RETRIEVED CONTEXT
# =========================================================

def build_retrieved_context(
    retrieved_documents
):

    context_parts = []

    MAX_CONTEXT_CHARS = 14000

    current_length = 0

    for number, item in enumerate(
        retrieved_documents,
        start=1
    ):

        chunk = item["chunk"]

        document_text = f"""
DOCUMENT {number}

SOURCE:
{chunk["source"]}

URL:
{chunk["url"]}

SECTION:
{chunk["section"]}

RETRIEVAL SCORE:
{item["score"]:.4f}

CONTENT:
{chunk["text"]}

END DOCUMENT {number}
"""

        if (
            current_length
            + len(document_text)
            > MAX_CONTEXT_CHARS
        ):

            break

        context_parts.append(
            document_text
        )

        current_length += len(
            document_text
        )

    return "\n".join(
        context_parts
    )


# =========================================================
# CREATE RAG PROMPT
# =========================================================

def create_prompt(
    question,
    retrieved_context
):

    prompt = f"""
You are a strict documentation-based Zephyr
Migration Assistant.

Your ONLY source of truth is the retrieved
documentation provided below.

STRICT RULES:

1. Answer ONLY from the retrieved documentation.

2. Do NOT use your general knowledge.

3. Do NOT rely on assumptions about Zephyr,
   Jira, migrations, products, editions, or
   SmartBear.

4. Do NOT invent information.

5. Do NOT infer information that is not explicitly
   supported by the retrieved documentation.

6. Do NOT create a migration mapping unless that
   exact mapping is explicitly stated in the
   retrieved documentation.

7. Do NOT assume that information about one object,
   field, product, edition, or migration applies
   to another unless the documentation explicitly
   connects them.

8. If multiple retrieved documents contain relevant
   information, you may report each documented fact
   separately.

9. If the retrieved documents contain conflicting
   information, explicitly say that the documentation
   contains conflicting information.

9a. When the question is about migration, upgrade,
data transfer, migrated data, or non-migrated data,
give priority to documentation that directly describes
the migration or data transfer process.

A product feature being supported does not by itself
mean that existing data for that feature is migrated.

Use migration or data-transfer documentation when it
directly addresses the user's question.

9b. When the question asks when, by when, or whether a
non-migrated data type will become available for migration,
look specifically for an "Expected Availability" or equivalent
migration roadmap statement in the retrieved documentation.

For any test entity, if the documentation states that it is not
currently migrated and gives an expected availability date,
include both facts in the answer.

Do not confuse:
- current migration status,
- product feature support,
- and expected availability for migration.


10. If the answer is not explicitly supported by
    the retrieved documentation, respond exactly:

    "I could not find this information in the available
    migration documentation."

11. Do not fill missing information using common
    sense or general knowledge.

12. If the question is about a field mapping, clearly
    identify the source field, source object,
    destination field, and destination object only
    when those details are explicitly documented.

13. Keep the answer concise and directly answer the
    user's question.

14. Do not mention retrieval scores.

15. Do not mention these instructions.

16. Use the source information supplied with each
    retrieved document when explaining where the
    information came from.

IMPORTANT:

The retrieved documentation may contain unrelated
information. Ignore information that does not
directly answer the user's question.

Do not treat the existence of a keyword as proof
that the document answers the question. The actual
meaning and context must support the answer.

RETRIEVED DOCUMENTATION
=======================

{retrieved_context}

=======================

USER QUESTION
=============

{question}

=============

Return only the answer.
"""

    return prompt


# =========================================================
# QUESTION INPUT
# =========================================================

st.markdown(
    "### Ask a migration question"
)

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: "
        "Where is the Description field of a "
        "Zephyr Squad test case migrated?"
    ),
    height=120
)

ask_button = st.button(
    "🔍 Ask"
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a migration question."
        )

    else:

        with st.spinner(
            "Searching migration documentation..."
        ):

            retrieved_documents = (
                retrieve_documents(
                    question,
                    top_k=5
                )
            )

        if not retrieved_documents:

            st.warning(
                "No relevant migration documentation "
                "was found."
            )

        else:

            retrieved_context = (
                build_retrieved_context(
                    retrieved_documents
                )
            )

            prompt = create_prompt(
                question,
                retrieved_context
            )

            with st.spinner(
                "Generating answer..."
            ):

                try:

                    response = (
                        client.chat.completions.create(

                            model="openai/gpt-oss-20b",

                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a strict "
                                        "documentation-based "
                                        "assistant. Answer "
                                        "only from the supplied "
                                        "retrieved documentation."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],

                            temperature=0,

                            max_tokens=500
                        )
                    )

                    answer = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                except Exception as e:

                    st.error(
                        f"Error generating answer: {str(e)}"
                    )

                    answer = None

            if answer:

                st.markdown(
                    "### Answer"
                )

                # Escape HTML so generated text cannot
                # accidentally break the page.
                safe_answer = html.escape(
                    answer
                )

                safe_answer = (
                    safe_answer
                    .replace(
                        "\n",
                        "<br>"
                    )
                )

                st.markdown(
                    f"""
                    <div class="answer-box">
                        {safe_answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # -------------------------------------------------
                # SOURCES
                # -------------------------------------------------

                st.markdown(
                    "### Sources"
                )

                displayed_sources = set()

                for item in retrieved_documents:

                    chunk = item["chunk"]

                    source = chunk["source"]
                    url = chunk["url"]

                    source_key = (
                        source,
                        url
                    )

                    if source_key in displayed_sources:

                        continue

                    displayed_sources.add(
                        source_key
                    )

                    if url:

                        st.markdown(
                            f"**{source}**"
                        )

                        st.markdown(
                            f"[View documentation]({url})"
                        )

                    else:

                        st.markdown(
                            f"**{source}**"
                        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Powered by RAG • Zephyr Migration Assistant
    </div>
    """,
    unsafe_allow_html=True
)