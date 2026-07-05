# PDF Q&A Chatbot

A document question-answering assistant that takes a PDF, extracts text, splits it into chunks, generates embeddings, and answers user questions using retrieved context.

Built with Google Gemini (via OpenAI-compatible API) and NumPy. No LangChain, no vector databases.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:

```
GOOGLE_API_KEY=your_key_here
```

Get one at https://aistudio.google.com/apikey

## Usage

```bash
# Run the Streamlit Web UI
streamlit run app.py

# Or use the CLI
python main.py path/to/document.pdf
```

Options:
- `--chunk-size N` — characters per chunk (default 500)
- `--overlap N` — overlap between chunks (default 50)
- `--top-k N` — number of chunks to retrieve (default 3)
- `--show-chunks` — show which chunks were retrieved

## Project Structure

```
pdf_loader.py    — extract text from PDF pages
chunker.py       — split text into overlapping chunks
embeddings.py    — generate embeddings via Gemini API
retriever.py     — cosine similarity search with NumPy
llm_client.py    — send context + question to Gemini LLM
app.py           — Streamlit Web UI entry point
main.py          — CLI entry point, ties everything together
```

## How It Works

1. Load PDF and extract text from all pages using `pypdf`
2. Split the text into ~500 character chunks with 50 char overlap
3. Generate embeddings for all chunks using `gemini-embedding-001`
4. For each question:
   - Embed the question
   - Compute cosine similarity (NumPy) against all chunk embeddings
   - Take top-3 most similar chunks
   - Send those chunks + question to `gemini-2.5-flash`
   - Print the answer
5. Conversation history (last 3 turns) is included for follow-up questions
