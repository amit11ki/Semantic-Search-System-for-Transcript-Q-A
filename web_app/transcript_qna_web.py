import os
import re
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

import google.generativeai as genai

# =============================
# 1) GEMINI CONFIGURATION
# =============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENABLED = False # Default to False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_ENABLED = True
    except Exception as e:
        st.warning(f"Gemini Pro API configuration failed: {e}. Gemini features will be disabled.")
else:
    # This warning will now appear in the sidebar if Gemini mode is selected without a key
    pass

# =============================
# 2) TRANSCRIPT PARSING UTILITIES
# =============================
def extract_timestamp(line: str):
    m = re.match(r'^\[(\d{2}:\d{2})(?::\d{2})?\s*-\s*(\d{2}:\d{2})(?::\d{2})?\]', line)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def strip_timestamp(line: str):
    return re.sub(r'^\[\d{2}:\d{2}(?::\d{2})?\s*-\s*\d{2}:\d{2}(?::\d{2})?\]\s*', '', line)

def chunk_transcript(lines: list[str], lines_per_chunk: int):
    """
    Convert raw transcript lines into chunks of `lines_per_chunk` lines each.
    Returns a list of dicts: { id, timestamp, text }.
    """
    chunks = []
    n = len(lines)
    for i in range(0, n, lines_per_chunk):
        group = lines[i : i + lines_per_chunk]
        if not group:
            continue

        start_ts, _ = extract_timestamp(group[0])
        _, end_ts = extract_timestamp(group[-1])
        # Allow chunks even if some lines don't have perfect timestamps,
        # as long as the first and last do for the chunk range.
        if start_ts is None: # Try to find first valid timestamp if first line missisng
            for l_idx, l_val in enumerate(group):
                start_ts_cand, _ = extract_timestamp(l_val)
                if start_ts_cand:
                    start_ts = start_ts_cand
                    break
        if end_ts is None: # Try to find last valid timestamp
             for l_idx in range(len(group) -1, -1, -1):
                _, end_ts_cand = extract_timestamp(group[l_idx])
                if end_ts_cand:
                    end_ts = end_ts_cand
                    break

        if start_ts is None or end_ts is None: # If still no valid start/end for chunk, skip
            # st.warning(f"Skipping a chunk because start/end timestamp couldn't be derived: {' '.join(group)}")
            continue

        texts = [strip_timestamp(l).strip() for l in group if strip_timestamp(l).strip()]
        combined_text = " ".join(texts)
        if not combined_text:
            continue

        chunks.append({
            "id": len(chunks),
            "timestamp": f"[{start_ts} - {end_ts}]",
            "text": combined_text,
        })
    return chunks

@st.cache_data
def load_and_chunk_in_memory(raw_lines: tuple[str, ...], lines_per_chunk: int):
    """
    Given a tuple of raw lines and lines_per_chunk, return chunked list.
    Caches based on raw_lines and lines_per_chunk.
    """
    return chunk_transcript(list(raw_lines), lines_per_chunk)

# =============================
# 3) TF-IDF SEARCH
# =============================
def tfidf_search(chunks: list[dict], query: str, top_k: int):
    texts = [c["text"] for c in chunks]
    if not texts:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[::-1][:top_k] # Get top_k indices
    # Filter out results with zero similarity
    return [(chunks[i]["timestamp"], chunks[i]["text"], sims[i]) for i in top_indices if sims[i] > 0.01]


# =============================
# 4) HUGGING FACE SEARCH (SEMANTIC SEARCH)
# =============================
@st.cache_resource # Cache the model resource
def load_hf_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load SentenceTransformer model: {e}")
        return None

def huggingface_search(chunks: list[dict], query: str, top_k: int):
    model = load_hf_model()
    if model is None:
        return []
    texts = [c["text"] for c in chunks]
    if not texts:
        return []
    try:
        corpus_emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        query_emb = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        hits = util.semantic_search(query_emb, corpus_emb, top_k=top_k)[0] # Get the first list of hits
        # Filter out results with low or zero similarity if desired (e.g. score > 0.1)
        return [
            (chunks[hit["corpus_id"]]["timestamp"], chunks[hit["corpus_id"]]["text"], hit["score"])
            for hit in hits if hit["score"] > 0.1 # Example threshold
        ]
    except Exception as e:
        st.error(f"Error during Hugging Face semantic search: {e}")
        return []

# =============================
# 5) GEMINI EMBEDDING & SEARCH (SEMANTIC SEARCH + GENERATIVE Q&A)
# =============================
@st.cache_data # Cache computed embeddings
def compute_gemini_embeddings(chunk_texts: tuple[str, ...]):
    if not GEMINI_ENABLED:
        # st.warning("Gemini is not enabled. Cannot compute embeddings.") # Already handled by GEMINI_ENABLED checks
        return None
    try:
        # Limit batch size if necessary, though embed_content handles individual texts
        embeddings = []
        for text_batch in chunk_texts: # Assuming chunk_texts is already batched if needed, or process one by one
            res = genai.embed_content(
                model="models/embedding-001", # Corrected model name
                content=text_batch, # This should be a single text or a list of texts based on API
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.append(np.array(res["embedding"]))
        return embeddings
    except Exception as e:
        st.error(f"Error generating Gemini chunk embeddings: {e}")
        return None

def get_query_embedding_gemini(query: str):
    if not GEMINI_ENABLED:
        return None
    try:
        res = genai.embed_content(
            model="models/embedding-001", # Corrected model name
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        return np.array(res["embedding"])
    except Exception as e:
        st.error(f"Error generating Gemini query embedding: {e}")
        return None

def gemini_search(chunks: list[dict], chunk_embeddings: list[np.ndarray], query: str, top_k: int):
    if not GEMINI_ENABLED or chunk_embeddings is None or not chunks:
        return []

    q_emb = get_query_embedding_gemini(query)
    if q_emb is None:
        return []

    q_vec = q_emb.reshape(1, -1) # Ensure query embedding is 2D for cosine_similarity
    sims = []

    for i, c_emb in enumerate(chunk_embeddings):
        if c_emb is not None:
            # Ensure chunk embedding is also 2D for cosine_similarity
            c_vec = c_emb.reshape(1, -1)
            score = cosine_similarity(q_vec, c_vec)[0][0]
            sims.append((i, score))

    sims.sort(key=lambda x: x[1], reverse=True)
    results = []
    count = 0
    for idx, score in sims:
        if score > 0.1 and count < top_k: # Example threshold
            results.append((chunks[idx]["timestamp"], chunks[idx]["text"], score))
            count += 1
    return results

def generate_answer_with_gemini(query: str, context_chunks: list[tuple[str, str, float]]):
    if not GEMINI_ENABLED or not context_chunks:
        return None
    try:
        # Prepare context, including only text for the prompt
        context_for_prompt = "\n".join([f"{i+1}. Timestamp: {ts}\n   Text: {text}" for i, (ts, text, score) in enumerate(context_chunks)])

        prompt = f"""You are a helpful AI assistant. Based *only* on the following transcript excerpts, provide a concise answer to the question.
If the information is not present in the excerpts, state that the information is not found in the provided context.
Do not make up information. Maximum 2-3 sentences.

Provided Transcript Excerpts:
{context_for_prompt}

Question: {query}

Answer:"""
        # Using a model that supports function calling and is good for Q&A if available, like gemini-1.5-flash or pro
        model = genai.GenerativeModel("gemini-1.5-flash-latest") # or "gemini-pro"
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        st.error(f"Gemini answer generation error: {e}")
        return "Sorry, I encountered an error while generating the answer."


# =============================
# 6) STREAMLIT UI LAYOUT
# =============================
st.set_page_config(page_title="Transcript Q&A Search Engine", layout="wide", initial_sidebar_state="expanded")
st.title("📄 Transcript Q&A Search Engine")
st.caption("Upload a timestamped transcript and ask questions. Explore results using keyword search or advanced semantic search with AI models.")

# --- Sidebar controls ---
st.sidebar.header("⚙️ Controls & Settings")
lines_per_chunk = st.sidebar.slider(
    "Lines per chunk for processing",
    min_value=1, max_value=15, value=5, step=1,
    help="Number of consecutive transcript lines to group into a single searchable 'chunk'. Larger chunks provide more context but might be less precise."
)
top_k = st.sidebar.slider(
    "Number of top chunks to display",
    min_value=1, max_value=10, value=3, step=1,
    help="How many of the most relevant transcript chunks to retrieve and display."
)

search_modes = ["TF-IDF (Keyword Search)", "Sentence Transformer (Semantic Search)", "Gemini AI (Semantic Search & Q&A)"]
selected_mode = st.sidebar.selectbox(
    "Choose Search Mode:",
    search_modes,
    help=(
        "TF-IDF: Fast keyword-based search.\n"
        "Sentence Transformer: Understands meaning and context (semantic).\n"
        "Gemini AI: Semantic search plus a generated answer from the context."
    )
)

if selected_mode == "Gemini AI (Semantic Search & Q&A)" and not GEMINI_ENABLED:
    st.sidebar.error("Gemini AI mode selected, but the API key is not configured or failed to initialize. Please set the GEMINI_API_KEY environment variable. Falling back to limited functionality if possible, or this mode may not work.")
elif not GEMINI_ENABLED and any("Gemini" in mode for mode in search_modes):
     st.sidebar.warning("GEMINI_API_KEY not found or invalid. 'Gemini AI' mode will be unavailable.")


# --- Main area: file uploader + query input ---
uploaded_file = st.file_uploader("📤 Upload your transcript file (.txt format)", type=["txt"])

if uploaded_file:
    raw_bytes = uploaded_file.read()
    try:
        # More robust decoding attempts
        try:
            content_lines = raw_bytes.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            content_lines = raw_bytes.decode("latin-1").splitlines() # Common fallback
    except UnicodeDecodeError:
        st.error("Error: Could not decode the file. Please ensure it's a plain text file (UTF-8 or Latin-1 encoded).")
        st.stop() # Stop execution if file can't be read

    if not content_lines:
        st.warning("The uploaded file appears to be empty.")
        st.stop()

    # Chunk transcript in memory (cached by raw_lines tuple + lines_per_chunk)
    chunks = load_and_chunk_in_memory(tuple(content_lines), lines_per_chunk)

    if not chunks:
        st.error("No valid text chunks could be extracted from the transcript. Please check the file format and content. Timestamps like [MM:SS - MM:SS] are expected for grouping.")
        st.stop()

    st.success(f"✅ Transcript loaded successfully! ({len(chunks)} chunks created). Ready for your questions.")
    user_query = st.text_input("💬 Enter your question about the transcript:", placeholder="e.g., What was said about project alpha?")

    if st.button("🔍 Search Transcript"):
        if not user_query.strip():
            st.warning("Please type a question before searching.")
        else:
            results = []
            st.markdown("---") # Separator before results
            st.subheader(f"Search Results using: {selected_mode}")

            if selected_mode == "TF-IDF (Keyword Search)":
                results = tfidf_search(chunks, user_query, top_k=top_k)
            elif selected_mode == "Sentence Transformer (Semantic Search)":
                results = huggingface_search(chunks, user_query, top_k=top_k)
            elif selected_mode == "Gemini AI (Semantic Search & Q&A)":
                if not GEMINI_ENABLED:
                    st.error("Gemini AI mode is unavailable. Please configure the GEMINI_API_KEY.")
                else:
                    chunk_texts_for_embedding = tuple(c["text"] for c in chunks)
                    chunk_embeddings = compute_gemini_embeddings(chunk_texts_for_embedding)
                    if chunk_embeddings:
                        results = gemini_search(chunks, chunk_embeddings, user_query, top_k=top_k)
                    else:
                        st.error("Could not compute embeddings for Gemini search.")

            # Display results
            if not results:
                st.info("ℹ️ No relevant chunks found for your question with the current settings.")
            else:
                st.markdown(f"**Top {len(results)} relevant chunks:**")
                for i, (ts, txt, score) in enumerate(results, 1):
                    with st.expander(f"{i}. Timestamp: {ts} (Similarity: {score:.2f})"):
                        st.write(txt)

                if selected_mode == "Gemini AI (Semantic Search & Q&A)" and results and GEMINI_ENABLED:
                    st.markdown("---")
                    st.subheader("💡 AI Generated Answer (via Gemini)")
                    with st.spinner("Gemini is thinking..."):
                        answer = generate_answer_with_gemini(user_query, results)
                    if answer:
                        st.success(f"**Answer:** {answer}")
                        st.caption("Source timestamps: " + ", ".join([res[0] for res in results]))
                    else:
                        st.warning("Gemini was unable to generate an answer based on the retrieved chunks, or an error occurred.")
else:
    st.info("Welcome! Please upload a transcript file to begin.")


# --- Footer Section ---
st.markdown("---")
st.markdown("""
### About the Search & Q&A Models

This application utilizes different techniques to search your transcript and help answer your questions:

* **TF-IDF (Keyword Search):**
    * This method uses Term Frequency-Inverse Document Frequency, a classical statistical measure.
    * It evaluates how relevant a word is to a document in a collection of documents (here, your transcript chunks).
    * It's effective for finding chunks containing specific keywords or phrases you enter.

* **Sentence Transformer (Semantic Search - `all-MiniLM-L6-v2`):**
    * This mode employs the `all-MiniLM-L6-v2` model, a highly efficient sentence embedding model from the Hugging Face Sentence Transformers library.
    * It converts your text chunks and your query into numerical representations (embeddings) that capture semantic meaning.
    * The search then finds chunks whose meanings are closest to your query's meaning, going beyond simple keyword matching to understand context and intent.

* **Gemini AI (Semantic Search & Q&A - Google):**
    * **Embedding Model (`models/embedding-001`):** Google's advanced embedding model is used to generate dense vector embeddings for both the transcript chunks and your query. This enables powerful semantic search, similar to the Sentence Transformer, by understanding the underlying meaning.
    * **Generative Model (`gemini-1.5-flash-latest`):** After the most relevant chunks are retrieved using semantic search, this state-of-the-art generative AI model from Google synthesizes a concise, direct answer to your question. The answer is based *only* on the information present in those retrieved transcript excerpts.

**Important Considerations:**
* **Semantic Search Quality:** The effectiveness of semantic search depends on the quality of the embeddings and the nature of your query and transcript content.
* **Generated Answers:** Answers from Gemini AI are derived solely from the retrieved transcript segments. If the information isn't in those segments, the model should indicate that.
* **API Keys:** "Gemini AI" mode requires a `GEMINI_API_KEY` environment variable to be correctly configured to access Google's services.
""")
st.markdown("<hr style='margin-top: 2em; margin-bottom: 1em;'>", unsafe_allow_html=True)
st.caption("Transcript Q&A Engine | Built with Streamlit & ❤️")