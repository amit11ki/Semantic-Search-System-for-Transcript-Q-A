# transcript.py

import sys
import re
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

import google.generativeai as genai

# ----------------------------
# CONFIGURE GEMINI API KEY (unchanged)
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDtySa9vyC_l9EmFof-_0UJbBmM_QVzdrU")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_ENABLED = True
except Exception as e:
    print(f"Warning: Gemini not configured properly ({e}). 'llm1' option will not work.")
    GEMINI_ENABLED = False

# ----------------------------
# 1) PARSE + CHUNK TRANSCRIPT
# ----------------------------
def extract_timestamp(line):
    m = re.match(r'^\[(\d{2}:\d{2})(?::\d{2})?\s*-\s*(\d{2}:\d{2})(?::\d{2})?\]', line)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def strip_timestamp(line):
    return re.sub(r'^\[\d{2}:\d{2}(?::\d{2})?\s*-\s*\d{2}:\d{2}(?::\d{2})?\]\s*', '', line)

def chunk_transcript(lines, lines_per_chunk=5):
    chunks = []
    n = len(lines)
    for i in range(0, n, lines_per_chunk):
        group = lines[i: i + lines_per_chunk]
        if not group:
            continue

        start_ts, _ = extract_timestamp(group[0])
        _, end_ts = extract_timestamp(group[-1])
        if start_ts is None or end_ts is None:
            continue

        texts = [strip_timestamp(l).strip() for l in group if strip_timestamp(l).strip()]
        combined_text = " ".join(texts)
        if not combined_text:
            continue

        chunks.append({
            "id": len(chunks),
            "timestamp": f"[{start_ts} - {end_ts}]",
            "text": combined_text,
            "embedding": None  # will be filled by precompute_gemini_embeddings()
        })
    return chunks

def load_and_chunk(transcript_path, lines_per_chunk=5):
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            raw_lines = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        print(f"Error: Transcript file '{transcript_path}' not found.")
        sys.exit(1)

    chunks = chunk_transcript(raw_lines, lines_per_chunk=lines_per_chunk)
    if not chunks:
        print("Warning: No valid chunks found in transcript.")
    return chunks

# ----------------------------
# 2) TF-IDF SEARCH
# ----------------------------
def tfidf_search(chunks, query, top_k=3):
    texts = [chunk["text"] for chunk in chunks]
    if not texts:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = sims.argsort()[::-1][:top_k]
    return [(chunks[i]["timestamp"], chunks[i]["text"]) for i in top_indices if sims[i] > 0]

# ----------------------------
# 3) HUGGING FACE (llm2)
# ----------------------------
_HF_MODEL = None
def load_hf_model():
    global _HF_MODEL
    if _HF_MODEL is None:
        _HF_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _HF_MODEL

def huggingface_search(chunks, query, top_k=3):
    model = load_hf_model()
    texts = [chunk["text"] for chunk in chunks]
    if not texts:
        return []

    corpus_embeddings = model.encode(texts, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

    return [
        (chunks[hit["corpus_id"]]["timestamp"], chunks[hit["corpus_id"]]["text"])
        for hit in hits
        if hit["score"] > 0
    ]

# ----------------------------
# 4) GEMINI (llm1): PRECOMPUTE EMBEDDINGS ONCE
# ----------------------------
def get_embeddings_gemini(texts_list, task_type="RETRIEVAL_DOCUMENT"):
    embeddings = []
    try:
        for text in texts_list:
            res = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type=task_type
            )
            embeddings.append(res['embedding'])
    except Exception as e:
        print(f"Error generating Gemini embeddings: {e}")
        return None
    return embeddings

def get_embedding_for_query_gemini(query):
    try:
        res = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        return res['embedding']
    except Exception as e:
        print(f"Error generating Gemini query embedding: {e}")
        return None

def precompute_gemini_embeddings(chunks):
    """
    Compute one embedding per chunk and store it in chunk['embedding'].
    Returns False on failure.
    """
    texts = [chunk["text"] for chunk in chunks]
    chunk_embs = get_embeddings_gemini(texts)
    if chunk_embs is None or len(chunk_embs) != len(texts):
        return False
    for i, emb in enumerate(chunk_embs):
        chunks[i]["embedding"] = np.array(emb)
    return True

def gemini_search(chunks, query, top_k=3):
    query_emb = get_embedding_for_query_gemini(query)
    if query_emb is None:
        return []

    query_emb_np = np.array(query_emb).reshape(1, -1)
    sims = []
    for chunk in chunks:
        if chunk["embedding"] is not None:
            sim = cosine_similarity(
                query_emb_np,
                chunk["embedding"].reshape(1, -1)
            )[0][0]
            sims.append((chunk, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return [
        (chunk["timestamp"], chunk["text"])
        for chunk, score in sims[:top_k]
        if score > 0
    ]

def generate_answer_with_gemini(question, context_chunks):
    try:
        context = "\n".join([f"{i+1}. {text}" for i, (_, text) in enumerate(context_chunks)])
        prompt = f"""Answer this question using only the information provided below. Be concise and direct (2-3 sentences max).

Information:
{context}

Question: {question}

Answer:"""
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        print(f"Error generating answer with Gemini: {e}")
        return None

# ----------------------------
# 5) CLI INTERFACE
# ----------------------------
def print_usage_and_exit():
    print("Usage: python transcript.py transcript.txt [tfidf|llm1|llm2]")
    sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print_usage_and_exit()

    transcript_path = sys.argv[1]
    mode = sys.argv[2].lower()
    valid_modes = {"tfidf", "llm1", "llm2"}
    if mode not in valid_modes:
        print_usage_and_exit()

    # 1) Load & chunk transcript
    chunks = load_and_chunk(transcript_path, lines_per_chunk=5)
    if not chunks:
        print("No chunks to search. Exiting.")
        sys.exit(0)

    # 2) If Gemini mode, precompute embeddings exactly once
    if mode == "llm1":
        if not GEMINI_ENABLED:
            print("Gemini (llm1) is not configured. Exiting.")
            sys.exit(1)
        success = precompute_gemini_embeddings(chunks)
        if not success:
            print("Failed to generate Gemini embeddings. Exiting.")
            sys.exit(1)

    print("\nTranscript loaded. You can now ask questions.")

    # 3) Enter query loop
    while True:
        try:
            query = input(
                "\nAsk a question\n"
                "(or press 8 to exit, or type 'switch to tfidf', 'switch to llm1', or 'switch to llm2')\n\n> "
            ).strip().lower()

            if query == "8":
                print("Goodbye!")
                break
            elif query.startswith("switch to "):
                new_mode = query.replace("switch to ", "").strip()
                if new_mode in valid_modes:
                    mode = new_mode
                    print(f"\nSwitched to mode: {mode.upper()}")
                    # If switching into llm1, precompute embeddings if not done
                    if mode == "llm1" and chunks and chunks[0]["embedding"] is None:
                        if not precompute_gemini_embeddings(chunks):
                            print("Failed to generate Gemini embeddings after switching. Exiting.")
                            sys.exit(1)
                else:
                    print("Invalid mode. Use 'tfidf', 'llm1', or 'llm2'.")
                continue
            elif not query:
                print("Please enter a question.")
                continue

            # 4) Run appropriate search
            if mode == "tfidf":
                results = tfidf_search(chunks, query, top_k=3)
            elif mode == "llm2":
                results = huggingface_search(chunks, query, top_k=3)
            elif mode == "llm1":
                results = gemini_search(chunks, query, top_k=3)
            else:
                print("Unsupported mode.")
                continue

            if not results:
                print("No relevant chunks found.")
                continue

            # 5) Display top chunks
            print("\nTop relevant chunks:")
            for i, (timestamp, text) in enumerate(results, 1):
                print(f"{i}. {timestamp}  {text}")

            # 6) If Gemini, also generate a concise answer
            if mode == "llm1":
                print("\nGenerating answer with Gemini...")
                answer = generate_answer_with_gemini(query, results)
                if answer:
                    print(f"\nAnswer: {answer}")
                    print(f"\nSources: {', '.join([ts for ts, _ in results])}")
                else:
                    print("\n(☹️) Gemini could not generate an answer. Showing chunks only above.")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
