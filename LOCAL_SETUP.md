# Local Configuration Guide

These steps configure and run the Zephyr Cloud Migration Assistant on Windows.

## Prerequisites

- Windows PowerShell
- Python 3.10 or newer
- A Groq API key
- Internet access on first run so the embedding model and SmartBear pages can be loaded

Run all commands from the project directory:

```powershell
cd "C:\Users\mamat\Documents\Zoom\AWS Monitoring Tools\rag-agent"
```

## 1. Create a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, either run the following once for your user account or use the activate script from Command Prompt:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. Install dependencies

The repository does not currently include a requirements file. Install the packages imported by the application and utility scripts:

```powershell
pip install streamlit truststore requests numpy beautifulsoup4 python-dotenv groq sentence-transformers chromadb
```

On the first use, `sentence-transformers` downloads the `all-MiniLM-L6-v2` model. That download can take a few minutes.

## 3. Configure the API key

Create a file named `.env` in the project root:

```dotenv
GROQ_API_KEY=replace_with_your_groq_api_key
```

Do not add quotes unless the key itself requires them, and do not commit this file. `.gitignore` already excludes `.env`.

The app stops at startup with `GROQ_API_KEY is not configured.` when this variable is missing.

## 4. Start the local server

This is a Streamlit application, so start it with Streamlit rather than plain Python:

```powershell
streamlit run app.py
```

Streamlit prints a local address, normally:

```text
http://localhost:8501
```

To use another port:

```powershell
streamlit run app.py --server.port 8502
```

Stop the server with `Ctrl+C` in the terminal.

## 5. Verify the application

1. Open the local URL in a browser.
2. Confirm the Zephyr header and logo appear.
3. Submit a migration question.
4. Check that an answer and source sections are displayed.
5. Use **Refresh Knowledge Base** after changing a file under `documents/`.

The app fetches SmartBear documentation at runtime. If a page is unavailable, the remaining local and remote sources are still used.

## Knowledge base files

Place local knowledge files in `documents/` as UTF-8 `.txt` files. The web app discovers these files automatically. Empty files and files with another extension are ignored.

The current UI uses these configured remote pages:

- Zephyr editions feature comparison
- Zephyr Squad to Zephyr upgrade guide
- Zephyr data transfer guide

## Optional Chroma workflow

The main Streamlit app creates embeddings in memory. The separate Chroma workflow can be run from the project root when a persistent vector database is needed:

```powershell
python create_vector_db.py
python retrieve.py
```

The database is written to `chroma_db/`. Run these scripts only after the dependencies and embedding model are installed.

## Troubleshooting

### `python app.py` exits immediately or fails

Use `streamlit run app.py`. `app.py` is a Streamlit entry point, not a regular command-line program.

### `GROQ_API_KEY is not configured.`

Confirm that `.env` is in the same directory as `app.py`, the variable name is exactly `GROQ_API_KEY`, and Streamlit was started from the project root.

### SSL or certificate errors

The app calls `truststore.inject_into_ssl()` before making network requests. If your organization uses a proxy or custom certificate, configure that in the local Python environment before starting the app.

### Remote documentation does not load

Check internet access and the SmartBear URLs in `app.py`. Local files under `documents/` remain available even when a remote page cannot be fetched.
