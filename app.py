import os
import streamlit as st
from pdf_loader import load_pdf
from chunker import split_into_chunks
from embeddings import init_embedding_client, embed_chunks, embed_query
from retriever import retrieve_top_k
from llm_client import init_llm_client, answer_question

# Initialize Streamlit page config
st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄", layout="centered")

st.title("📄 PDF Q&A Assistant")
st.write("Upload a PDF document and ask questions about its content.")

# Initialize session state for storing embeddings and history
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    # History for the LLM context: list of (question, answer) tuples
    st.session_state.history = []

# Sidebar for file upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily so our pdf_loader can read it
        temp_pdf_path = "temp_uploaded.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Only process if we haven't already processed this file
        if st.button("Process Document"):
            with st.spinner("Processing PDF and generating embeddings..."):
                try:
                    # 1. Init clients
                    embed_client = init_embedding_client()
                    st.session_state.embed_client = embed_client
                    st.session_state.llm_client = init_llm_client()
                    
                    # 2. Extract and chunk
                    full_text = load_pdf(temp_pdf_path)
                    chunks = split_into_chunks(full_text, chunk_size=500, overlap=50)
                    
                    # 3. Embed chunks
                    chunk_embeddings = embed_chunks(embed_client, chunks)
                    
                    # Save to session state
                    st.session_state.chunks = chunks
                    st.session_state.chunk_embeddings = chunk_embeddings
                    
                    # Clear chat history when a new document is processed
                    st.session_state.messages = []
                    st.session_state.history = []
                    
                    st.success(f"Document processed successfully! Generated {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)

# Main chat interface
if st.session_state.chunks is not None:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "context" in message:
                with st.expander("View Retrieved Context"):
                    st.markdown(message["context"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # 1. Embed query
                    query_emb = embed_query(st.session_state.embed_client, prompt)
                    
                    # 2. Retrieve top-k chunks
                    result = retrieve_top_k(
                        chunks=st.session_state.chunks,
                        chunk_embeddings=st.session_state.chunk_embeddings,
                        query_embedding=query_emb,
                        top_k=3
                    )
                    
                    # Prepare context string for expander display
                    context_display = ""
                    for i, (chunk, score, idx) in enumerate(zip(result.chunks, result.scores, result.indices)):
                        context_display += f"**Chunk #{idx} (Score: {score:.4f})**\n\n{chunk}\n\n---\n\n"
                    
                    # 3. Get answer from LLM (using history bonus feature!)
                    answer = answer_question(
                        client=st.session_state.llm_client,
                        question=prompt,
                        context_chunks=result.chunks,
                        history=st.session_state.history
                    )
                    
                    # Display assistant response
                    st.markdown(answer)
                    
                    with st.expander("View Retrieved Context"):
                        st.markdown(context_display)
                        
                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "context": context_display}
                    )
                    
                    # Update the backend history format (list of tuples)
                    st.session_state.history.append((prompt, answer))
                    
                except Exception as e:
                    st.error(f"Error getting answer: {e}")
else:
    st.info("👈 Please upload and process a PDF document in the sidebar to begin.")
